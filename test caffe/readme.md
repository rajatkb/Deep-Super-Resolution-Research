Refer to this post https://www.reddit.com/r/deepdream/comments/3du9hl/experimental_prebuilt_cudaenabled_caffe_binary/  
And the official caffe installation page  

1.  Define your network in a prototext  
2.  Use caffe.Net('text file' , caffe.TEST) to define the network
3.  you have the entire network initialised at this point. yay !!!!!
4.  i.e after this you want the data look at caffe.net().blob it stores all the data  
    of all the layers with their respective name as keys  these data are actually THE ACTUALL COMPUTED DATA
5.  caffe.net().params hold the layers wise weights  
    params['name'][0] is the filter weights and params['name'][1] is the bias weights  

6.  im = np.array(Image.open('examples/images/cat_gray.jpg'))
    im_input = im[np.newaxis, np.newaxis, :, :]
    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input
    construct the images in shape of (m , channel , height , width)
    rest all happy and done also make sure to have ur input blobs in size of the image
7.  net.save('mymodel.caffemodel') to save the model
8.  Rest is in code the solver included is an Adam solver there is more available out in the wild
    https://github.com/BVLC/caffe/wiki/Solver-Prototxt usefull link for solvertext  

9.  And read this the caffe-tools it will save your damn life https://github.com/davidstutz/caffe-tools  
    in case you have issues with LMDB like i have. Also caffe can work with formats. It is just slow  
    and this is the defacto
