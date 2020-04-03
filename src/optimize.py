def init_best_spikestags(model_info):
    for lay_info in model_info:
        if (not isinstance(lay_info['lay'],nn.Conv2d)):
            continue
        filter_nb = lay_info['lay'].out_channels
        lay_info['best_spikes'] = torch.zeros(filter_nb,TOP_AVG_SIGN)
        lay_info['best_tags'] = torch.ones(filter_nb,TOP_AVG_SIGN) * -1
init_best_spikestags(model_info)

batch_size = 20
my_dataset = ImageFolder('./data/tinyimagenet/train/',transform=preprocess)
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=False)
for j,(datas,labels) in enumerate(train_loader):
        clear_output(wait=True)
        print("Progression:{:.2f} %".format(j/len(train_loader) * 100))
        for i,lay_info in enumerate(model_info):
            datas = lay_info['lay'](datas)
            if (not isinstance(lay_info['lay'],nn.Conv2d)):
                continue
            nb_filters = lay_info['lay'].out_channels
            
            #tags: Numberfilter x Batchsize
            tags = torch.range(j * batch_size,(j+1)* batch_size-1).repeat(nb_filters).view(nb_filters,batch_size)
            
            #datas: Batchsize x Numberfilter x Nout x Nout
            batch_size = datas.size(0)
            filter_nb = datas.size(1)
                        
            #spikes: Batch_size x Numberfilter
            spikes = datas.view([filter_nb,batch_size,-1]).mean(dim = 2)
            
            #spikes: Numberfilter x Batch_size 
            #spikes = spikes.transpose(1,0)
            
            #tot_tags: Numberfilter x (Batch_size + best_size)
            tot_spikes = torch.cat((spikes,lay_info['best_spikes']),dim=1)
            
            #tot_tags: Numberfilter x (Batch_size + best_size)
            tot_tags = torch.cat((tags,lay_info['best_tags']),dim=1)
            
            tot_spikes,sort_indx = tot_spikes.sort()
            
            #sorts the tot_tags inplace
            tot_tags = torch.cat([row[idx].unsqueeze(0) for idx,row in zip(sort_indx,tot_tags)])
            
            #take the best_size last entries
            lay_info['best_spikes'] = tot_spikes[:,-TOP_AVG_SIGN:]
            lay_info['best_tags'] = tot_tags[:,-TOP_AVG_SIGN:] 