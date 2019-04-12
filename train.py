import yaml
from docopt import docopt
import os,time,random, rawpy
import tensorflow as tf
import numpy as np
import net as net
import utils as utils
import loss as losses
from PIL import Image
from shutil import copyfile
from tensorflow.core.protobuf import config_pb2

# For parallel image reading
from multiprocessing import Process, Pool, Queue

is_debug = True
if loss_type == 'contextual':
    w_align = 0
    w_smooth = 0.
    w_cont = 1
    w_patch = 2
    w_spatial = 0.5
elif loss_type == 'combine':
    w_align = 100.
    w_cont = 1
    w_patch = 1.5
    w_smooth = 10
    w_spatial = 0.0
elif loss_type == 'align':
    w_align = 1
    w_cont = 1
    w_patch = 0.
    w_smooth = 0
    w_spatial = 0.0

boundary_tol: 16
raw_tol = 4
if mode == 'train':
    img_sz = 512
else:
    img_sz = 512
raw_sz = img_sz/up_ratio/2 + raw_tol*2

if not os.path.isdir("%s"%(task)):
    os.makedirs("%s"%(task))
    copyfile('./train.py', './%s/train_%s.py'%(task, time.strftime('%b-%d-%Y_%H%M', time.localtime())))
    copyfile('./utils.py', './%s/utils_%s.py'%(task, time.strftime('%b-%d-%Y_%H%M', time.localtime())))
    copyfile('./model/net.py', './%s/net_%s.py'%(task, time.strftime('%b-%d-%Y_%H%M', time.localtime())))
    copyfile('./loss.py', './%s/loss_%s.py'%(task, time.strftime('%b-%d-%Y_%H%M', time.localtime())))
    with open('./%s/loss.txt'%(task), 'w') as floss:
        floss.write('%s\n'%task)
if not os.path.isdir("/home/xuanerzh/tmp_%s"%(substr)):
    os.makedirs("/home/xuanerzh/tmp_%s"%(substr))

### YUV2RGB
def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.multiply(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(tf.shape(temp), dtype=tf.float32))
    temp = tf.minimum(temp, tf.multiply(
        tf.ones(tf.shape(temp), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp

def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp

def gamma(rgb, gamma=2.2):
    rgb=tf.clip_by_value(rgb, 0., 1000)
    return rgb ** (1/gamma)

def degamma(rgb, gamma=2.2):
    rgb=tf.clip_by_value(rgb, 0., 1000)
    return rgb ** (gamma)

def prepare_train_data(raw_path):
    tar_h_val = img_sz #np.random.randint(64, 72)*(2*up_ratio)
    tar_w_val = img_sz #np.random.randint(64, 72)*(2*up_ratio)
    rgb_path = raw_path.replace(".ARW",".JPG")
    
    # Load raw-jpg image pairs
    if up_ratio == 4:
        id_shift = 3
    elif up_ratio == 8:
        id_shift = 5
    elif id_shift == 2:
        id_shift = 2
    input_dict = utils.read_input_2x(raw_path, rgb_path, id_shift=id_shift)
    if input_dict is None:
        print("input_dict is None ... for %s"%(raw_path))
        return None
    processed_dict = utils.prepare_input(input_dict, up_ratio=up_ratio, is_pack=True)
    if processed_dict is None:
        print('Invalid processed_dict %s'%(raw_path))
        return None
    input_raw_img_orig,target_rgb_img_orig = processed_dict['input_raw'], processed_dict['tar_rgb']
    if input_raw_img_orig is None or target_rgb_img_orig is None:
        print('Invalid input raw or rgb for %s'%(raw_path))
        return None

    # prepare input to pre-align
    row, col = input_raw_img_orig.shape[0:2]
    target_rgb_img_orig, transformed_corner = utils.post_process_rgb(target_rgb_img_orig,
        (int(col*2*up_ratio),int(row*2*up_ratio)), processed_dict['tform'])
    input_raw_img_orig = input_raw_img_orig[int(transformed_corner['minw']/(2*up_ratio)):int(transformed_corner['maxw']/(2*up_ratio)),
        int(transformed_corner['minh']/(2*up_ratio)):int(transformed_corner['maxh']/(2*up_ratio)),:]
    # pre-align to camera output
    # target_rgb_img_orig, transformed_corner = utils.post_process_rgb(target_rgb_img_orig,
    #     (int(col*2*up_ratio),int(row*2*up_ratio)), processed_dict['camera_tform'])
    # input_raw_img_orig = input_raw_img_orig[int(transformed_corner['minw']/(2*up_ratio)):int(transformed_corner['maxw']/(2*up_ratio)),
    #     int(transformed_corner['minh']/(2*up_ratio)):int(transformed_corner['maxh']/(2*up_ratio)),:]
    
    cropped_raw, cropped_rgb = utils.crop_pair(input_raw_img_orig, target_rgb_img_orig, 
        croph=tar_h_val, cropw=tar_w_val, tol=tol, raw_tol=raw_tol, ratio=up_ratio, type='central')
    
    # remove white balance
    out_wb = input_dict['tar_wb']
    cropped_rgb[...,0] /= np.power(out_wb[0,0],1/2.2)
    cropped_rgb[...,1] /= np.power(out_wb[0,1],1/2.2)
    cropped_rgb[...,2] /= np.power(out_wb[0,3],1/2.2)

    cropped = {}
    cropped['cropped_raw'] = cropped_raw
    cropped['cropped_rgb'] = cropped_rgb
    cropped['input_dict'] = input_dict

    return cropped
    
def read_images(input_queue, files):
    for file in files:
        input_queue.put(prepare_train_data(file))

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/hourglass.yaml"
    checkpoints = args["<checkpoints>"]
    with open(config_file, "r") as f:
        c = yaml.load(f)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]

    with tf.variable_scope(tf.get_variable_scope()):
        input_raw=tf.placeholder(tf.float32,shape=[batch_size,raw_sz,raw_sz,num_in_ch], name="input_raw")
        out_rgb = net.SRResnet(input_raw, num_out_ch, up_ratio=up_ratio, reuse=False, up_type=upsample_type, is_training=True)

        if raw_tol != 0:
            out_rgb = out_rgb[:,int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),
                int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),:]  # add a small offset to deal with boudary case

        objDict = {}
        lossDict = {}
        objDict['out_rgb'] = out_rgb

        target_rgb=tf.placeholder(tf.float32,shape=[batch_size,img_sz+tol*2,img_sz+tol*2,3], name="target_rgb")
        tar_lum = rgb2yuv(target_rgb)[...,0,tf.newaxis]
        objDict['tar_lum'] = tar_lum

        loss_context = tf.constant(0.)
        loss_unalign = tf.constant(0.)
        if loss_type == 'align' or loss_type == 'combine':
            loss_unalign, target_translated=losses.compute_unalign_loss(out_rgb, target_rgb,
                tol=tol, losstype=align_loss_type, stride=stride)
            loss_unalign *= w_align
            if align_loss_type == 'percep':
                lossDict['percep'] = loss_unalign
            # loss_unalign += tf.reduce_mean(tf.abs(out_rgb-target_translated)) * 500
            tf.summary.scalar('loss_unalign', loss_unalign)
            objDict['target_translated'] = target_translated

        if loss_type == 'contextual' or loss_type == 'combine':
            _, target_translated=losses.compute_unalign_loss(out_rgb, target_rgb,
                tol=tol, losstype=align_loss_type, stride=stride)
            target_translated = tf.reshape(target_translated, [batch_size,img_sz,img_sz, num_out_ch])
            loss_context,loss_argmax = losses.compute_contextual_loss(out_rgb, target_translated, w_spatial=w_spatial)
            loss_context_patch,_ = losses.compute_patch_contextual_loss(out_rgb, target_translated, 
                patch_sz=1, rates=1, w_spatial=w_spatial)
            tf.summary.scalar('loss_context', loss_context)
            tf.summary.scalar('loss_context_patch', loss_context_patch)
        
        lossDict['l1'] = tf.reduce_mean(tf.abs(out_rgb-target_translated))
        lossDict['loss_context'] = loss_context
        lossDict['l1_unalign'] = loss_unalign
        lossDict['total'] = loss_context * w_cont + loss_context_patch * w_patch + loss_unalign * w_align

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(0.0001, global_step, decay_step, decay_rate)
    if opt_type == 'Momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=0.00005, momentum=0.99).minimize(lossDict['total'],
            global_step=global_step,
            var_list=[var for var in tf.trainable_variables()])
    elif opt_type == 'Adam':
        # learning_rate=tf.placeholder(tf.float32)
        opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lossDict['total'],
            global_step=global_step,
            var_list=[var for var in tf.trainable_variables()])
    
    incr_global_step = tf.assign(global_step, global_step + 1)
    merged = tf.summary.merge_all()
    saver=tf.train.Saver(max_to_keep=10)

    ###################################### Session
    sess=tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('%s/log'%(task), sess.graph)
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])

    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state("%s"%(restore_path))
    epoch_offset = 0

    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name='%s/model.ckpt'%(restore_path), tensor_name='', all_tensors=False)

    print("contain checkpoint: ", ckpt)
    if ckpt and continue_training:
        saver_restore.restore(sess,ckpt.model_checkpoint_path)
        try:
            epoch_offset = tf.train.get_checkpoint_state("%s"%(restore_path)).all_model_checkpoint_paths[-2]
            epoch_offset = int(os.path.basename(os.path.dirname(epoch_offset)))
        except Exception as exception:
            epoch_offset = 0
        print('loaded %s with epoch offset %d'%(ckpt.model_checkpoint_path, epoch_offset))

    train_input_paths=utils.read_paths(train_root, type=file_type)
    train_input_paths=train_input_paths
    num_train_per_epoch=len(train_input_paths)
    train_input_paths = train_input_paths*maxepoch
    num_train = len(train_input_paths)
    print("Number of training images: ", num_train_per_epoch)
    print("Total %d raw images" % (num_train_per_epoch))
    all_loss=np.zeros(num_train, dtype=float)
    cnt=0
    epoch = 0

    train_input_paths_permute = list(np.random.permutation(train_input_paths))
    num_process = 8
    input_queue = Queue(16)
    for i in range(num_process):
        producer = Process(target=read_images, args=(input_queue, train_input_paths_permute[i::num_process]))
        producer.start()

    for id in range(num_train):
        if id % num_train_per_epoch == 0:
            epoch += 1
            print("Processing epoch %d"%(epoch))
        
        if os.path.isdir("%s/%04d"%(task,epoch)):
            continue
        
        cropped = input_queue.get()

        if cropped is None:
            continue
        
        cropped_raw = cropped['cropped_raw']
        cropped_rgb = cropped['cropped_rgb']
        input_dict = cropped['input_dict']

        if cropped_raw is None or cropped_rgb is None:
            continue

        print("Processed image shapes: ", cropped_raw.shape, cropped_rgb.shape, cropped_raw.min(), cropped_raw.max())
        if (cropped_raw.shape[0] - raw_tol*2) * 2 * up_ratio + tol * 2 != cropped_rgb.shape[0]:
            print("xxxxx Wrong cropped shapes.")
            continue
        if (cropped_raw.shape[1] - raw_tol*2) * 2 * up_ratio + tol * 2 != cropped_rgb.shape[1]:
            print("xxxxx Wrong cropped shapes.")
            continue
        
        target_rgb_img = np.expand_dims(cropped_rgb, 0)
        input_raw_img = np.expand_dims(cropped_raw, 0)
        # print("RGB image range: ",target_rgb_img.min(), target_rgb_img.max())
        
        fetch_list=[opt,learning_rate,global_step,merged,objDict,lossDict,loss_argmax]
        st=time.time()
        _, out_lr, out_global_step, out_summary,out_objDict,out_lossDict,out_arg=sess.run(fetch_list,feed_dict=
            {input_raw:input_raw_img,
            target_rgb:target_rgb_img})
            # tar_shape:[1,tar_h_val,tar_w_val,num_out_ch]})
            # learning_rate:min(1e-5*np.power(1.1,epoch-1),1e-5)}) #learning_rate:min(1e-5*np.power(1.1,epoch-1),1e-4)})
        all_loss[id]=out_lossDict["total"]

        cnt+=1
        
        print("%s --- lr: %.5f --- Global_step: %d" % (task,out_lr, out_global_step))
        print("Iter: %d %d || loss: (t) %.4f (l1) %.4f (l_cont) %.4f (l_smooth) %.4f || mean: %.4f || time: %.2f"%
            (epoch,cnt,out_lossDict["total"],
                out_lossDict["l1"],
                out_lossDict["loss_context"],
                out_lossDict['smooth'],
                np.mean(all_loss[np.where(all_loss)]),
                time.time()-st))
        
        if cnt % save_summary_freq == 0:
            writer.add_summary(out_summary, cnt)
        if is_debug and (cnt-1) % save_img_freq == 0:
            with open('./%s/loss.txt'%(task), 'a') as floss:
                floss.write('loss for Iter %d is %.4f\n'%(cnt, out_lossDict["total"]))
            cropped_raw_reshaped = utils.reshape_back_raw(cropped_raw)
            cropped_raw_rgb = utils.write_raw(cropped_raw_reshaped, input_dict['src_path_raw'])
            # cropped_raw_rgb = cropped_raw_rgb[:cropped_raw_reshaped.shape[0],:cropped_raw_reshaped.shape[1]]
            cropped_raw_rgb = cropped_raw_rgb[raw_tol*2:cropped_raw_reshaped.shape[0]-raw_tol*2,
                raw_tol*2:cropped_raw_reshaped.shape[1]-raw_tol*2]

            input_rgb_cropped = Image.fromarray(cropped_raw_rgb)
            input_rgb_cropped = input_rgb_cropped.resize((int(input_rgb_cropped.width * up_ratio),
                int(input_rgb_cropped.height * up_ratio)), Image.ANTIALIAS)
            input_rgb_cropped.save("/home/xuanerzh/tmp_%s/input_rgb_cropped_%d_%d.png"%(substr, epoch, cnt))

            output_rgb = utils.apply_gamma(
                utils.clipped(np.squeeze(out_objDict["out_rgb"][0,...])),is_apply=False)*255
            
            src_raw = Image.fromarray(np.uint8(utils.reshape_back_raw(utils.apply_gamma(input_raw_img[0,...]))*255))
            tartget_rgb = Image.fromarray(np.uint8(utils.clipped(
                utils.apply_gamma(target_rgb_img[0,...], is_apply=False))*255))
            Image.fromarray(np.uint8(output_rgb)).save("/home/xuanerzh/tmp_%s/out_rgb_%d_%d.png"%(substr, epoch, cnt))
            src_raw.save("/home/xuanerzh/tmp_%s/src_raw_%d_%d.png"%(substr, epoch, cnt))
            tartget_rgb.save("/home/xuanerzh/tmp_%s/tar_rgb_%d_%d.png"%(substr, epoch, cnt))
            # if loss_type == 'align' or loss_type == 'combine':
            #     gt_match = Image.fromarray(np.uint8(utils.clipped(utils.apply_gamma(np.squeeze(
            #         out_objDict['target_translated']),is_apply=is_gt_gamma==False))*255))
            #     gt_match.save("/home/xuanerzh/tmp_%s/tar_rgb_match_%d_%d.png"%(substr, epoch, cnt))

            target_buffer = target_rgb_img

        if cnt % save_model_freq == 0:
            print("saving model ...")
            if not os.path.isdir("%s%s/%04d"%(save_root,task,epoch)):
                os.makedirs("%s%s/%04d"%(save_root,task,epoch))
            saver.save(sess,"%s%s/model.ckpt"%(save_root,task))
            saver.save(sess,"%s%s/%04d/model.ckpt"%(save_root,task,epoch))
            try:
                output_rgb = utils.clipped(np.squeeze(out_objDict["out_rgb"][0,...]))*255
                output_rgb = Image.fromarray(np.uint8(output_rgb))
                tartget_rgb = utils.clipped(utils.apply_gamma(target_buffer[0,...]))*255
                tartget_rgb = Image.fromarray(np.uint8(tartget_rgb))
                output_rgb.save("%s%s/%04d/out_rgb_%s.png"%(save_root,task,epoch,substr))
                tartget_rgb.save("%s%s/%04d/tar_rgb_%s.png"%(save_root,task,epoch,substr))
                if loss_type == 'align' or loss_type == 'combine':
                    gt_match = Image.fromarray(np.uint8(utils.clipped(utils.apply_gamma(np.squeeze(out_objDict['target_translated'])))*255))
                    gt_match.save("%s%s/%04d/tar_rgb_match_%s.png"%(save_root,task,epoch,substr))
            except Exception as exception:
                print("Failed to write image footprint. ;(")
        floss.close()

if __name__ == "__main__":
    main()