#train functions

parset_py() #call to parser

data_dirDoc = 'flowers' #flowers reference

dataload_create(data_dirDoc) #dataloader call


dataload_create(data_dir) #loads data and returns dataloader to use

classifier_create(model)# creates classifier

train(epochs, pre_trained_model)# trains model

def save_check(pre_trained_model)# saves checkpoint


#predict functions

parsep_py()# parser function

load_checkpoint()# loads checkpoint

process_image(image)# processes images in PIL

imshow(image, ax=None, title=None)# shows the image

predict(image_path, model, topk=5)# makes predictions from image

output()# outputs graph and 5 topk