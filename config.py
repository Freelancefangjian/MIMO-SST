class FLAGES(object):

    pan_size= 32
    
    ms_size=4
    
    
    num_spectrum=4
    
    ratio=4
    stride=2
    norm=True
    
    
    batch_size=32
    lr=0.0001
    decay_rate=0.99
    decay_step=10000
    
    img_path='G:\\OneDrive - njust.edu.cn\\Hyperspectral_Image_Fusion_Benchmarkx8\\CAVE\\MW-DAN\\data'
    data_path='G:\\OneDrive - njust.edu.cn\\Hyperspectral_Image_Fusion_Benchmarkx8\\CAVE\\MW-DAN\\data\\train\\train_qk.h5'
    
    is_pretrained=False
    
    iters=500000
    model_save_iters = 500
    valid_iters=10
