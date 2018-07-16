import os,time
import tensorflow as tf
import numpy as np
import model.net as net
import utils as utils
import loss as losses
from PIL import Image

is_debug = False
continue_training = True
mode = 'test'
task = 'restore_vanilla'
train_root = ['/home/xuanerzh/Downloads/zoom/dslr_10x_both/']
num_channels = 64
save_freq = 10
num_in_ch = 4
num_out_ch = 12
batch_size = 1
pw, ph = 480, 480
tw, th = 540, 540
file_type='RAW'
subfolder = 'dslr_10x_both'

with tf.variable_scope(tf.get_variable_scope()):
    input_raw=tf.placeholder(tf.float32,shape=[batch_size,None,None,num_in_ch])
    target_rgb=tf.placeholder(tf.float32,shape=[batch_size,None,None,3])
    
    out_rgb=net.build_unet(input_raw,
        channel=num_channels,
        input_channel=num_in_ch,
        output_channel=num_out_ch,
        reuse=False,
        num_layer=8)

    loss_context=losses.(prediction, target, pw, ph, tw, th, losstype='l1', align=False)
    loss_l1=losses.compute_l1_loss(out_rgb, target_rgb)

    objDict = {}
    lossDict = {}
    objDict['out_rgb'] = out_rgb
    lossDict['l1'] = loss_l1
    lossDict['context'] = loss_context
    loss_sum = sum(lossDict.values())
    lossDict['total'] = loss_sum

###################################### Session
sess=tf.Session()
opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_sum,
    var_list=[var for var in tf.trainable_variables()])
saver=tf.train.Saver(max_to_keep=10)
saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(task)
print("contain checkpoint: ", ckpt)
if ckpt and continue_training:
    print('loaded '+ ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)

train_input_paths=utils.read_paths(train_root, type=file_type)
maxepoch=100
num_train=len(train_input_paths)
print("Number of training images: ", num_train)
if mode == "train":
    print("Total %d raw images" % (len(train_input_paths)))
    all_loss=np.zeros(num_train, dtype=float)
    for epoch in range(1,maxepoch):
        print("Processing epoch %d"%epoch)
        input_raw_img=[None]*num_train
        target_rgb_img=[None]*num_train
        if os.path.isdir("%s/%04d"%(task,epoch)):
            continue
        cnt=0
        for id in np.random.permutation(num_train):
            if input_raw_img[id] is None:
                train_input_path2 = utils.read_input_2x(train_input_paths[id], train_input_paths[id].replace(subfolder, subfolder+'_process/rawpng'))
                if train_input_path2 is None:
                    continue
                input_raw_img[id],target_rgb_img[id] = utils.prepare_input(train_input_paths[id], train_input_path2)
                # print("Image:",train_input_paths[id], input_raw_img[id].shape, target_rgb_img[id].shape)

                if input_raw_img[id] is None or target_rgb_img[id] is None:
                    print('Invalid input raw or rgb for %s'%(train_input_paths[id]))
                    continue
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
                if is_debug and cnt % 5 == 0:
                    output_rgb = out_objDict["out_rgb"][0,...]*255
                    output_rgb = Image.fromarray(np.uint8(output_rgb))
                    tartget_rgb = target_rgb_img[id][0,...]*255
                    tartget_rgb = Image.fromarray(np.uint8(tartget_rgb))
                    output_rgb.save("/home/xuanerzh/tmp/out_rgb_%d.png"%(cnt))
                    tartget_rgb.save("/home/xuanerzh/tmp/tar_rgb_%d.png"%(cnt))
                target_buffer = target_rgb_img[id]
                input_raw_img[id]=1.
                target_rgb_img[id]=1.

        if epoch % save_freq == 0:
            if not os.path.isdir("%s/%04d"%(task,epoch)):
                os.makedirs("%s/%04d"%(task,epoch))
            saver.save(sess,"%s/model.ckpt"%task)
            saver.save(sess,"%s/%04d/model.ckpt"%(task,epoch))
            output_rgb = out_objDict["out_rgb"][0,...]*255
            output_rgb = Image.fromarray(np.uint8(output_rgb))
            tartget_rgb = target_buffer[0,...]*255
            tartget_rgb = Image.fromarray(np.uint8(tartget_rgb))
            output_rgb.save("%s/%04d/out_rgb.png"%(task,epoch))
            tartget_rgb.save("%s/%04d/tar_rgb.png"%(task,epoch))

else:
    test_folder = 'test'
    if not os.path.isdir("%s/%s"%(task, test_folder)):
        os.makedirs("%s/%s"%(task, test_folder))
    for id in np.random.permutation(num_train):
        print("Testing on %d th image"%(id))
        input_raw_img,target_rgb_img = utils.prepare_input(train_input_paths[id])
        fetch_list=[objDict,lossDict]
        out_objDict,out_lossDict=sess.run(fetch_list,feed_dict=
            {input_raw:input_raw_img,target_rgb:target_rgb_img})
        output_rgb = out_objDict["out_rgb"][0,...]*255
        output_rgb = Image.fromarray(np.uint8(output_rgb))
        tartget_rgb = target_rgb_img[0,...]*255
        tartget_rgb = Image.fromarray(np.uint8(tartget_rgb))
        output_rgb.save("%s/%s/%d_out_rgb.png"%(task,test_folder,id))
        tartget_rgb.save("%s/%s/%d_tar_rgb.png"%(task,test_folder,id))