def UNIT_TEST1():
    #test max slices
    layer_indx = 28
    filter_indx = 7
    top_indx = 0
    lay_info = model_info[layer_indx]
    filtr = lay_info["filters"][filter_indx]
    img_path = filtr["max_imgs"][top_indx]
    spike = filtr["max_spikes"][top_indx]
    print("Max spikes for layer {} filter {}:{}".format(layer_indx,filter_indx,filtr["max_spikes"]))
    print("Studied spike: {}".format(spike))

    test_image = preprocess(Image.open(img_path)).unsqueeze(0)
    for i,lay in enumerate(model_info):
        if i > layer_indx:
            break
        test_image = lay['lay'](test_image)
    assert(test_image.shape[1] == model_info[layer_indx]['lay'].out_channels)
    print("Retreated image max:{}".format(test_image[0][filter_indx].max().item()))

    row_size = test_image.shape[2]

    linindx = test_image[0][filter_indx].view(-1).max(0)[1].item()
    row = linindx // row_size
    col = linindx % row_size
    #needs to be equal to the 2 max above
    print("Check for recovered area {}".format(test_image[0][filter_indx][row,col]))
    area = lay_info['deproj'].chain(((row,row),(col,col)))
    print("Correct area:{}".format(area))
    print("Official area: {} ".format(filtr['max_slices'][top_indx]))

    fig,axes = plt.subplots(1,3,figsize=(10,20))
    test_image = preprocess(Image.open(img_path))
    axes[0].imshow(test_image.permute(1,2,0))

    ((x1,x2),(y1,y2)) = filtr['max_slices'][top_indx]
    cropped = test_image[:,x1:x2+1,y1:y2+1]
    axes[1].imshow(cropped.permute(1,2,0))

    test_image = preprocess(Image.open(img_path)).unsqueeze(0)
    ((x1,x2),(y1,y2)) = filtr['max_slices'][top_indx]
    artif_image = torch.zeros((1,3,224,224))
    artif_image[0,:,x1:x2+1,y1:y2+1] = test_image[0,:,x1:x2+1,y1:y2+1]
    axes[2].imshow(artif_image[0].permute(1,2,0))

    for i,lay in enumerate(model_info):
        if i > layer_indx:
            break
        artif_image = lay['lay'](artif_image)
    print("Recoverd max for the artificial image:{}".format(artif_image[0][filter_indx].max().item()))

def UNIT_TEST2():
    #test max spike
    layer_indx = 2
    filter_indx = 9
    top_indx = 2
    filtr = model_info[layer_indx]["filters"][filter_indx]
    img_path = filtr["max_imgs"][top_indx]
    spike = filtr["max_spikes"][top_indx]
    print("Spikes:{}".format(filtr["max_spikes"]))
    print("Current spike: {}".format(spike))
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=12, shuffle=True)

    my_dataset = ImageFolderWithPaths('./data/exsmallimagenet/',transform=preprocess)
    CLASS2INDX = my_dataset.class_to_idx
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=True)
    for j,(datas,labels,paths) in enumerate(train_loader):
        if j > 10:
            break
        for i,lay_info in enumerate(model_info):
            if i > layer_indx:
                break
            datas = lay_info['lay'](datas)
        print("Random max:{}".format(datas[0][filter_indx].max().item()))
        
def UNIT_TEST3():
    #test avg spike
    layer_indx = 2
    filter_indx = 9
    filtr = model_info[layer_indx]["filters"][filter_indx]
    img_path = filtr["fav_imgs"][0]
    spike = filtr["spikes"][0]
    print("Spikes:{}".format(filtr["spikes"]))
    print("Current spike: {}".format(spike))

    test_image = preprocess(Image.open(img_path)).unsqueeze(0)
    for i,lay_info in enumerate(model_info):
        if i > layer_indx:
            break
        test_image = lay_info['lay'](test_image)
    assert(test_image.shape[1] == model_info[layer_indx]['lay'].out_channels)
    print("Recoverd spike:{}".format(test_image[0][filter_indx].mean().item()))
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=12, shuffle=True)

    my_dataset = ImageFolderWithPaths('./data/exsmallimagenet/',transform=preprocess)
    CLASS2INDX = my_dataset.class_to_idx
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=True)
    for j,(datas,labels,paths) in enumerate(train_loader):
        if j > 10:
            break
        for i,lay_info in enumerate(model_info):
            if i > layer_indx:
                break
            datas = lay_info['lay'](datas)
        print("Random avg:{}".format(datas[0][filter_indx].mean().item()))
