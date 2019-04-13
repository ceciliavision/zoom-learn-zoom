import yaml
from PIL import Image
import tensorflow as tf
import glob, os
import numpy as np
import net as net
import utils as utils

def main():

    config_file = "config/inference.yaml"
    with open(config_file, "r") as f:
        config_file = yaml.load(f)


    # Hyper-parameters
    mode = config_file["mode"]
    device = config_file["device"]
    up_ratio = config_file["model"]["up_ratio"]
    num_in_ch = config_file["model"]["num_in_channel"]
    num_out_ch = config_file["model"]["num_out_channel"]
    file_type = config_file["model"]["file_type"]
    upsample_type = config_file["model"]["upsample_type"]
    
    save_root = config_file["io"]["save_root"]
    test_root = [config_file["io"]["test_root"]]
    task_folder = config_file["io"]["task_folder"]
    restore_path = config_file["io"]["restore_ckpt"]

    # Remove boundary pixels with artifacts
    raw_tol = 4
    if device == "sony":
        white_lv = 16383
        black_lv = 512
    elif device == "iphone":
        white_lv = 4367
        black_lv = 528

    with tf.variable_scope(tf.get_variable_scope()):
        input_raw=tf.placeholder(tf.float32,shape=[1,None,None,num_in_ch], name="input_raw")
        out_rgb = net.SRResnet(input_raw, num_out_ch, up_ratio=up_ratio, reuse=False, up_type=upsample_type, is_training=True)

        if raw_tol != 0:
            out_rgb = out_rgb[:,int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),
                int(raw_tol/2)*(up_ratio*4):-int(raw_tol/2)*(up_ratio*4),:]  # add a small offset to deal with boudary case

        objDict = {}
        objDict['out_rgb'] = out_rgb

    ###################################### Session
    sess=tf.Session()
    merged = tf.summary.merge_all()
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])

    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state("%s"%(restore_path))
    print("contain checkpoint: ", ckpt)
    if not ckpt:
        print("No checkpoint found.")
        exit()
    else:
        saver_restore.restore(sess,ckpt.model_checkpoint_path)

    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name='%s/model.ckpt'%(restore_path), tensor_name='', all_tensors=False)

    if mode == 'inference':
        test_input_paths=utils.read_paths(test_root, type=file_type)
        
        if not os.path.isdir("%s/%s"%(task_folder, mode)):
            os.makedirs("%s/%s"%(task_folder, mode))
        
        inference_path = None
        if inference_path is None:
            num_test = len(test_input_paths)
        else:
            num_test=1
            test_input_paths[0] = inference_path

        for id in range(num_test):
            inference_path = test_input_paths[id]
            rgb_path = inference_path.replace(".ARW",".JPG")

            if up_ratio == 4:
                id_shift = 3
            elif up_ratio == 8:
                id_shift = 5
            
            input_dict = utils.read_input_2x(inference_path, rgb_path, id_shift=id_shift, mode=mode)
            if input_dict is None:
                continue
            print("Inferencing on %d th image: %s"%(id, input_dict['src_path_raw']))
            fileid = int(os.path.basename(input_dict['src_path_raw']).split('.')[0])
            folderid = int(os.path.basename(os.path.dirname(input_dict['src_path_raw'])))
            
            processed_dict = utils.prepare_input(input_dict, up_ratio=up_ratio, mode='inference')
            input_raw_img_orig = processed_dict['input_raw']
            input_raw_orig = rawpy.imread(input_dict['src_path_raw'])
            # input_rgb_img_orig = input_raw_orig.postprocess(gamma=(1, 1),no_auto_bright=True,use_camera_wb=False,output_bps=8)
            # row, col = input_raw_img_orig.shape[0:2]
            # #tar_rgb = Image.open(os.path.dirname(input_dict['tar_path'])+'/'+os.path.basename(input_dict['tar_path'].split('.')[0]+'.png'))
            # #tar_rgb_orig = utils.crop_fov(np.array(tar_rgb), 1./input_dict['ratio_ref1'])
            # #target_rgb_img_orig, transformed_corner = utils.post_process_rgb(tar_rgb_orig,
            # #        (int(col*2*up_ratio),int(row*2*up_ratio)), processed_dict['tform'])
            # input_rgb_img_orig = utils.crop_fov(input_rgb_img_orig, 1./input_dict['ratio_ref2'])
            input_rgb_img_orig = processed_dict['input_rgb']
            if input_raw_img_orig is None:
                print('Invalid input raw or rgb for %s'%(inference_path))
                continue

            # prepare input to pre-align
            
            input_raw_img = np.expand_dims(input_raw_img_orig, 0)

            print("Processed image raw shapes: ", input_raw_img.shape)
            out_objDict=sess.run(objDict,feed_dict={input_raw:input_raw_img},
                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            
            print("Finish inference ... ")
            print("Stats: ", out_objDict["out_rgb"][0,...].mean(), out_objDict["out_rgb"][0,...].max(), out_objDict["out_rgb"][0,...].min())
            if not os.path.isdir("%s/%s/%d_%d"%(task_folder, mode, folderid, fileid)):
                os.makedirs("%s/%s/%d_%d"%(task_folder, mode, folderid, fileid))
            out_wb = input_dict['src_wb']
            print("white balance: ",out_wb)
            out_objDict["out_rgb"][0,...,0] *= np.power(out_wb[0,0],1/2.2)
            out_objDict["out_rgb"][0,...,1] *= np.power(out_wb[0,1],1/2.2)
            out_objDict["out_rgb"][0,...,2] *= np.power(out_wb[0,3],1/2.2)

            output_rgb = Image.fromarray(np.uint8(
                utils.apply_gamma(utils.clipped(np.squeeze(out_objDict["out_rgb"][0,...])),is_apply=False)*255))
            input_rgb = Image.fromarray(np.uint8(utils.apply_gamma(input_rgb_img_orig[int(raw_tol/2)*(4):-int(raw_tol/2)*(4),
                                        int(raw_tol/2)*(4):-int(raw_tol/2)*(4),:],is_apply=False)*255))
            input_rgb.save("%s/%s/%d_%d/input_rgb.png"%(task_folder,mode,folderid,fileid))
            input_rgb_naive = input_rgb.resize((int(input_rgb.width * up_ratio),
                int(input_rgb.height * up_ratio)), Image.BILINEAR)
            input_rgb_naive.save("%s/%s/%d_%d/input_rgb_naive.png"%(task_folder,mode,folderid,fileid))
            output_rgb.save("%s/%s/%d_%d/out_rgb.png"%(task_folder,mode,folderid,fileid))

    # Quick inference on a single image
    elif mode == 'inference_single':
        up_ratio_list = [8]
        inference_path = config_file["io"]['inference_path']
        for up_ratio in up_ratio_list:
            scale = 10.
            resize_ratio = up_ratio/scale

            if not os.path.isdir("%s/%s"%(task_folder, mode)):
                os.makedirs("%s/%s"%(task_folder, mode))

            id = '134-%s'%(up_ratio)
            if not os.path.isdir("%s/%s/%s"%(task_folder, mode, id)):
                os.makedirs("%s/%s/%s"%(task_folder, mode, id))

            wb_txt = os.path.dirname(inference_path)+'/wb.txt'
            out_wb = utils.read_wb(wb_txt, key=os.path.basename(inference_path).split('.')[0]+":")
            print("white balance: ",out_wb)

            input_bayer = utils.get_bayer(inference_path, black_lv, white_lv)
            input_raw_reshape = utils.reshape_raw(input_bayer)
            input_raw_img_orig = utils.crop_fov(input_raw_reshape, 1./up_ratio)

            rgb_camera_path = inference_path.replace(".ARW",".JPG")
            rgb_camera =  np.array(Image.open(rgb_camera_path))
            cropped_input_rgb = utils.crop_fov(rgb_camera, 1./up_ratio)
            cropped_input_rgb = utils.image_float(cropped_input_rgb)

            print("Testing on image : %s"%(inference_path), input_raw_img_orig.shape)

            input_raw_img = np.expand_dims(input_raw_img_orig, 0)
            out_objDict=sess.run(objDict,feed_dict={input_raw:input_raw_img})
            
            wb_rgb = out_objDict["out_rgb"][0,...]
            wb_rgb[...,0] *= np.power(out_wb[0,0],1/2.2)
            wb_rgb[...,1] *= np.power(out_wb[0,1],1/2.2)
            wb_rgb[...,2] *= np.power(out_wb[0,3],1/2.2)
            
            output_rgb = Image.fromarray(np.uint8(utils.clipped(wb_rgb)*255))
            output_rgb = output_rgb.resize((int(output_rgb.width * resize_ratio),
                int(output_rgb.height * resize_ratio)), Image.ANTIALIAS)
            output_rgb.save("%s/%s/%s/out_rgb.png"%(task_folder,mode,id))

            input_camera_rgb = Image.fromarray(np.uint8(utils.clipped(cropped_input_rgb)*255))
            input_camera_rgb.save("%s/%s/%s/input_rgb_camera_orig.png"%(task_folder,mode,id))
            input_camera_rgb_naive = input_camera_rgb.resize((int(input_camera_rgb.width * up_ratio),
                int(input_camera_rgb.height * up_ratio)), Image.ANTIALIAS)
            input_camera_rgb_naive.save("%s/%s/%s/input_rgb_camera_naive.png"%(task_folder,mode,id), compress_level=1)


if __name__ == "__main__":
    main()