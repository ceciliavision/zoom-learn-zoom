import os,time
import tensorflow as tf
import numpy as np
import model.net as net
import utils as utils
import loss as losses
from PIL import Image
from shutil import copyfile

is_debug = True
continue_training = True
mode = 'test'
task = 'restore_vanilla'
train_root = ['/home/xuanerzh/Downloads/zoom/dslr_10x_both/'] # path to raw files
subfolder = 'dslr_10x_both'
maxepoch = 100

num_channels = 64
save_freq = 2   # unit: epoch
save_img_freq = 200   #unit: step
num_in_ch = 4
num_out_ch = 48
batch_size = 1
tar_h, tar_w = 512, 512
tol = 32
up_ratio = 2
stride = 8
file_type='RAW'

if mode == 'train' and not continue_training:
    copyfile('./train.py', './%s/train.py'%(task))

def tf_crop_image(img, crop):
    return tf.image.crop_to_bounding_box(img, crop[0][0], crop[0][1], crop[0][2], crop[0][3])

with tf.variable_scope(tf.get_variable_scope()):
    input_raw=tf.placeholder(tf.float32,shape=[batch_size,None,None,num_in_ch])
    target_rgb=tf.placeholder(tf.float32,shape=[batch_size,None,None,3])
    # crop_box=tf.placeholder(tf.int32,shape=[batch_size,4])

    out_rgb=net.build_unet(input_raw,
        channel=num_channels,
        input_channel=num_in_ch,
        output_channel=num_out_ch,
        reuse=False,
        num_layer=5)

    objDict = {}
    objDict['out_rgb'] = out_rgb
    if mode == "train":
        loss_context=losses.compute_unalign_loss(out_rgb, target_rgb,
            tar_h=tar_h, tar_w=tar_w, tol=tol, losstype='percep', stride=stride)

        lossDict = {}
        lossDict['l1'] = loss_context
        # lossDict['context'] = loss_context
        loss_sum = sum(lossDict.values())
        lossDict['total'] = loss_sum

if mode == "train":
    opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_sum,
        var_list=[var for var in tf.trainable_variables()])

###################################### Session
sess=tf.Session()
saver=tf.train.Saver(max_to_keep=10)
saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(task)
epoch_offset = 0
print("contain checkpoint: ", ckpt)
if ckpt and continue_training:
    print('loaded '+ ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)
    epoch_offset = tf.train.get_checkpoint_state(task).all_model_checkpoint_paths[-2]

if mode == "train":
    train_input_paths=utils.read_paths(train_root, type=file_type)
    num_train=len(train_input_paths)
    print("Number of training images: ", num_train)
    print("Total %d raw images" % (len(train_input_paths)))
    all_loss=np.zeros(num_train, dtype=float)
    for epoch in range(1+epoch_offset, maxepoch+epoch_offset):
        print("Processing epoch %d"%(epoch))
        input_raw_img=[None]*num_train
        target_rgb_img=[None]*num_train
        if os.path.isdir("%s/%04d"%(task,epoch)):
            continue
        cnt=0
        for id in np.random.permutation(num_train):
            if input_raw_img[id] is None:
                rgb_path = os.path.dirname(train_input_paths[id]).replace(subfolder, subfolder+'_process') + "/rawpng/" + \
                    os.path.basename(train_input_paths[id]).replace(".ARW",".png")
                input_dict = utils.read_input_2x(train_input_paths[id], rgb_path)
                if input_dict is None:
                    continue
                processed_dict = utils.prepare_input(input_dict)
                input_raw_img_orig,target_rgb_img_orig = processed_dict['input_raw'], processed_dict['tar_rgb']
                # print("Read in image shapes:",train_input_paths[id], input_raw_img_orig.shape, target_rgb_img_orig.shape)
                if input_raw_img_orig is None or target_rgb_img_orig is None:
                    print('Invalid input raw or rgb for %s'%(train_input_paths[id]))
                    continue

                # prepare input to pre-align
                row, col = input_raw_img_orig.shape[0:2]
                target_rgb_img_orig, transformed_corner = utils.post_process_rgb(target_rgb_img_orig,
                    (int(col*2*up_ratio),int(row*2*up_ratio)), processed_dict['tform'])
                print(row, col, transformed_corner)
                input_raw_img_orig = input_raw_img_orig[int(transformed_corner['minw']/(2*up_ratio)):int(transformed_corner['maxw']/(2*up_ratio)),
                    int(transformed_corner['minh']/(2*up_ratio)):int(transformed_corner['maxh']/(2*up_ratio)),:]
                print(input_raw_img_orig.shape)
                cropped_raw, cropped_rgb = utils.crop_pair(input_raw_img_orig, target_rgb_img_orig, 
                    croph=tar_h, cropw=tar_w, tol=tol, ratio=up_ratio, type='central')
                target_rgb_img[id] = np.expand_dims(cropped_rgb, 0)
                input_raw_img[id] = np.expand_dims(cropped_raw, 0)

                # ratio = 4
                # input_raw_img_orig = Image.fromarray(input_raw_img_orig[0,...,0])
                # input_raw_img_orig = input_raw_img_orig.resize((int(input_raw_img_orig.width * ratio),
                #     int(input_raw_img_orig.height * ratio)), Image.ANTIALIAS)
                # input_raw_img_orig = np.expand_dims(input_raw_img_orig, 0)
                # input_raw_img_orig = np.expand_dims(input_raw_img_orig, 3)
                # input_raw_img[id] = input_raw_img_orig[:,transformed_corner['minw']:transformed_corner['maxw'],
                #     transformed_corner['minh']:transformed_corner['maxh'],:]

                # crop_box_input = np.array([transformed_corner['minw'], transformed_corner['minh'],
                #     transformed_corner['maxh']-transformed_corner['minh'], transformed_corner['maxw'] - transformed_corner['minw']])
                # crop_box_input = np.expand_dims(crop_box_input, 0).astype(np.int32)
                print("Processed image shapes: ", input_raw_img[id].shape, target_rgb_img[id].shape)

                file=os.path.splitext(os.path.basename(train_input_paths[id]))[0]
                fetch_list=[opt,objDict,lossDict]
                st=time.time()
                _,out_objDict,out_lossDict=sess.run(fetch_list,feed_dict=
                    {input_raw:input_raw_img[id],target_rgb:target_rgb_img[id]})
                all_loss[id]=out_lossDict["total"]

                cnt+=1
                
                print("iter: %d %d || loss: (t) %.4f (l1) %.4f || mean: %.4f || time: %.2f"%
                    (epoch,cnt,out_lossDict["total"],
                        out_lossDict["l1"],
                        np.mean(all_loss[np.where(all_loss)]),
                        time.time()-st))
                if is_debug and cnt % save_img_freq == 0:
                    output_rgb = out_objDict["out_rgb"][0,...]*255
                    src_raw = Image.fromarray(np.uint8(utils.apply_gamma(input_raw_img[id][0,...,0])*255))
                    tartget_rgb = Image.fromarray(np.uint8(utils.apply_gamma(target_rgb_img[id][0,...])*255))
                    Image.fromarray(np.uint8(output_rgb)).save("/home/xuanerzh/tmp/out_rgb_%d_%d.png"%(epoch, cnt))
                    src_raw.save("/home/xuanerzh/tmp/src_raw_%d_%d.png"%(epoch, cnt))
                    tartget_rgb.save("/home/xuanerzh/tmp/tar_rgb_%d_%d.png"%(epoch, cnt))
                target_buffer = target_rgb_img[id]
                input_raw_img[id]=1.
                target_rgb_img[id]=1.

        if epoch % save_freq == 0:
            if not os.path.isdir("%s/%04d"%(task,epoch)):
                os.makedirs("%s/%04d"%(task,epoch))
            saver.save(sess,"%s/model.ckpt"%task)
            saver.save(sess,"%s/%04d/model.ckpt"%(task,epoch))
            try:
                output_rgb = utils.apply_gamma(out_objDict["out_rgb"][0,...])*255
                output_rgb = Image.fromarray(np.uint8(output_rgb))
                tartget_rgb = utils.apply_gamma(target_buffer[0,...])*255
                tartget_rgb = Image.fromarray(np.uint8(tartget_rgb))
                output_rgb.save("%s/%04d/out_rgb.png"%(task,epoch))
                tartget_rgb.save("%s/%04d/tar_rgb.png"%(task,epoch))
            except:
                print("Failed to write image footprint. ;(")

else:
    test_input_paths=utils.read_paths(train_root, type=file_type)
    num_test=len(test_input_paths)
    test_folder = 'test'
    if not os.path.isdir("%s/%s"%(task, test_folder)):
        os.makedirs("%s/%s"%(task, test_folder))
    for id in np.random.permutation(num_test):
        print("Testing on %d th image : %s"%(id, test_input_paths[id]))
        rgb_path = os.path.dirname(test_input_paths[id]).replace(subfolder, subfolder+'_process') + "/rawpng/" + \
            os.path.basename(test_input_paths[id]).replace(".ARW",".png")
        print(rgb_path)
        input_dict = utils.read_input_2x(test_input_paths[id], rgb_path)
        if input_dict is None:
            continue
        processed_dict = utils.prepare_input(input_dict)
        input_raw_img_orig,target_rgb_img_orig = processed_dict['input_raw'], processed_dict['tar_rgb']
        input_rgb_img_orig = processed_dict['input_rgb']
        # print("Read in image shapes:",test_input_paths[id], input_raw_img_orig.shape, target_rgb_img_orig.shape)
        if input_raw_img_orig is None or target_rgb_img_orig is None:
            print('Invalid input raw or rgb for %s'%(test_input_paths[id]))
            continue

        # prepare input to pre-align
        row, col = input_raw_img_orig.shape[0:2]
        target_rgb_img_orig, transformed_corner = utils.post_process_rgb(target_rgb_img_orig,
            (int(col*2*up_ratio),int(row*2*up_ratio)), processed_dict['tform'])
        print(row, col, transformed_corner)
        input_raw_img_orig = input_raw_img_orig[int(transformed_corner['minw']/(2*up_ratio)):int(transformed_corner['maxw']/(2*up_ratio)),
            int(transformed_corner['minh']/(2*up_ratio)):int(transformed_corner['maxh']/(2*up_ratio)),:]
        
        input_rgb_img_orig = input_rgb_img_orig[int(transformed_corner['minw']/(up_ratio)):int(transformed_corner['maxw']/(up_ratio)),
            int(transformed_corner['minh']/(up_ratio)):int(transformed_corner['maxh']/(up_ratio))]
        print(input_raw_img_orig.shape, input_rgb_img_orig.shape)
        
        cropped_raw, cropped_rgb = utils.crop_pair(input_raw_img_orig, target_rgb_img_orig, 
            croph=tar_h, cropw=tar_w, tol=tol, ratio=up_ratio, type='central')
        target_rgb_img = np.expand_dims(cropped_rgb, 0)
        input_raw_img = np.expand_dims(cropped_raw, 0)

        fetch_list=objDict
        out_objDict=sess.run(fetch_list,feed_dict=
            {input_raw:input_raw_img,target_rgb:target_rgb_img})
        output_rgb = Image.fromarray(np.uint8(utils.apply_gamma(out_objDict["out_rgb"][0,...])*255))
        gt_rgb = Image.fromarray(np.uint8(utils.apply_gamma(cropped_rgb)*255))
        input_rgb = Image.fromarray(np.uint8(utils.apply_gamma(input_rgb_img_orig)*255))
        input_rgb.save("%s/%s/%d_input_rgb.png"%(task,test_folder,id))
        output_rgb.save("%s/%s/%d_out_rgb.png"%(task,test_folder,id))
        gt_rgb.save("%s/%s/%d_tar_rgb.png"%(task,test_folder,id))