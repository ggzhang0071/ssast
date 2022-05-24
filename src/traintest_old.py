# ! /usr/bi
# Time: 20/3/22 12:57 AM
# Author: Gege Zhang


from operator import index
from sklearn.semi_supervised import LabelSpreading
import torch
import torch.nn as nn
from  torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utilities import *
import numpy as np
import os, json
import pandas as pd
import pickle
from torch.utils.tensorboard import SummaryWriter
import logging
logging.basicConfig(level=logging.WARNING)

writer=SummaryWriter("test real child ed data")

def train(audio_model, train_loader,val_loader,test_loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model,nn.DataParallel):
        audio_model=nn.DataParallel(audio_model)   # get the model from the dataparallel
    audio_model=audio_model.to(device)

    # setup the optimizer
    trainables=[p for p in audio_model.parameters() if p.requires_grad]
    logging.info("Total parameter number is:{:.3f} million".format(sum(p.numel() for p in audio_model.parameters())/1e6))
    logging.info("Total trainable parameter number is:{:.3f} million".format(sum(p.numel() for p in trainables)/1e6))
    optimizer=torch.optim.Adam(trainables, args.lr,weight_decay=5e-7,betas=(0.95, 0.999))

    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    if args.dataset=="audioset":
        if  len(train_loader.dataset)> 2e5:
            logging.info("schedule for full dataset is used")
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5],gamma=0.5,last_epoch=-1)
        else:
            logging.info("schedule for small dataset is used")
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5,last_epoch=-1)
        main_metrics="mAP"
        loss_fn=nn.BCEWithLogitsLoss()
        warmup=True
    elif args.dataset=="esc50":
        logging.info("schedule for esc50 is used")
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,list(range(5,26)), gamma=0.85)
        main_metrics="acc"
        loss_fn=nn.CrossEntropyLoss()
        warmup=False
    else:
        raise Exception("dataset not supported, please choose the audioset or esc50")
    logging.info('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    args.loss_fn=loss_fn


    exp_dir=args.exp_dir
    scaler=GradScaler()
    pbar=tqdm(range(1,args.n_epoches+1))
    print("start training...")
    # Initialize the all the statstics we want to keep track of
    train_loss_meter=AverageMeter()
    result = np.zeros([args.n_epoches, 12])
    audio_model.train()
    for epoch in pbar: 
        pbar.set_description("Epoch {}".format(epoch))
        for audio_input, label in (train_loader):
            torch.cuda.empty_cache()
            N=audio_input.size(0)
            audio_input=audio_input.to(device,non_blocking=True)
            label=label.to(device,non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                output=audio_model(audio_input)
                if isinstance(args.loss_fn,nn.CrossEntropyLoss):
                    loss=args.loss_fn(output, torch.argmax(label.long(), axis=1))
                else:
                    loss=args.loss_fn(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # record loss
            train_loss_meter.update(loss.item(), N)
        train_loss=train_loss_meter.avg
        train_loss_meter.reset()

        print("start validation")
        stats,valid_loss=validate(audio_model, val_loader, args,epoch)
        writer.add_scalar("train/loss", train_loss, epoch)


        mAP=np.mean([stat["AP"] for stat in stats])
        mAUC=np.mean([stat["auc"] for stat in stats])
        mf1_score=np.mean([stat["f1_score"] for stat in stats])
        acc=stats[0]["acc"]

        writer.add_scalar("valid/loss", valid_loss, epoch)
        writer.add_scalar("valid/mAP", mAP, epoch)
        writer.add_scalar("valid/acc", acc, epoch)
        writer.add_scalar("valid/f1_score", mf1_score, epoch)

        # ensemble results
        cum_stats=validate_ensemble(args, epoch)
        cum_mAP=np.mean([stat["AP"] for stat in cum_stats])
        cum_mAUC=np.mean([stat["auc"] for stat in cum_stats])
        cum_mf1_score=np.mean([stat["f1_score"] for stat in cum_stats])
        cum_acc=cum_stats[0]["acc"]


        middle_ps=[stat["precisions"][int(len(stat['precisions'])/2)]  for stat in stats]
        middle_rs=[stat["recalls"][int(len(stat['recalls'])/2)]  for stat in stats]
        average_precision=np.mean(middle_ps)
        average_recall=np.mean(middle_rs)
        #average_f1_score=np.mean(middle_f1s)

        if main_metrics == 'mAP':
            logging.info("mAP: {:.6f}".format(mAP))
        else:
            logging.info("acc: {:.6f}".format(acc))

        logging.info("AUC: {:.6f}".format(mAUC))
        logging.info("Avg Precision: {:.6f}".format(average_precision))
        logging.info("Avg Recall: {:.6f}".format(average_recall))
        logging.info("d_prime: {:.6f}".format(d_prime(mAUC)))
        logging.info("train_loss: {:.6f}".format(train_loss))
        logging.info("valid_loss: {:.6f}".format(valid_loss))

        writer.add_scalar("valid/cum_mAP", cum_mAP, epoch)
        writer.add_scalar("valid/cum_acc", cum_acc, epoch)
        writer.add_scalar("valid/cum_f1_score", cum_mf1_score, epoch)


        if main_metrics == 'mAP':
            result[epoch-1, :] = [mAP, mAUC,mf1_score, average_precision, average_recall, d_prime(mAUC), train_loss, valid_loss, cum_mAP, cum_mAUC,cum_mf1_score, optimizer.param_groups[0]['lr']]
        else:
            result[epoch-1,:]=[acc, mAUC, mf1_score,average_precision, average_recall, d_prime(mAUC), train_loss, valid_loss, cum_acc, cum_mAUC, cum_mf1_score,optimizer.param_groups[0]['lr']]
        
        np.savetxt(exp_dir+"/result.txt", result, fmt="%.4f",delimiter=",")
        print("valiladtion result saved")

        # save the best model and optimizer
        if mAP>best_mAP:
            best_mAP=mAP
            if main_metrics=="mAP":
                best_epoch=epoch  
                torch.save(audio_model.state_dict(), exp_dir+"/models/best_audio_model.pth")
                torch.save(optimizer.state_dict(), exp_dir+"/models/best_optimizer_state.pth")

        torch.save(audio_model.state_dict(), exp_dir+"/models/audio_model_{}.pth".format(epoch))

        if  len(train_loader.dataset)>2e5:
            torch.save(optimizer.state_dict(), exp_dir+"/models/optimizer_state_{}.pth".format(epoch))
        
        
        scheduler.step()
        print("epoch-{} lr: {}".format(epoch, optimizer.param_groups[0]['lr']))
        
        with open(exp_dir+"/stats_"+str(epoch)+".pickle", "wb") as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    writer.close()

    get_test_labels(audio_model,test_loader, args)   


    if args.dataset=="audioset":
        if len(train_loader.dataset)>2e5:
            stats=validate_wa(audio_model, val_loader, args,1,5)
        else:
            #stats=validate_wa(audio_model, test_loader, args, 6, 25)
            stats=validate_wa(audio_model, val_loader, args,1,2)
        mAP=np.mean([stat["AP"] for stat in stats])
        mAUC=np.mean([stat["auc"] for stat in stats])
        middle_ps=[stat["precisions"][int(len(stat['precisions'])/2)]  for stat in stats]
        middle_rs=[stat["recalls"][int(len(stat['recalls'])/2)]  for stat in stats]
        average_precision=np.mean(middle_ps)
        average_recall=np.mean(middle_rs)
        wa_result=[mAP,mAUC,average_precision,average_recall,d_prime(mAUC)]
        logging.info("-------------Training finished-------------")
        logging.info("weighted average model results")
        logging.info("mAP: {:.6f}".format(mAP))
        logging.info("AUC: {:.6f}".format(mAUC))
        logging.info("Avg Precision: {:.6f}".format(average_precision))
        logging.info("Avg Recall: {:.6f}".format(average_recall))
        logging.info("d_prime: {:.6f}".format(d_prime(mAUC)))
        logging.info("train_loss: {:.6f}".format(train_loss))
        logging.info("valid_loss: {:.6f}".format(valid_loss))
        np.savetxt(exp_dir+"/wa_result.txt", wa_result, fmt="%.4f",delimiter=",")



def validate(audio_model, val_loader,args,epoch):
    valid_loss_meter=AverageMeter()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model,nn.DataParallel):
        audio_model=nn.DataParallel(audio_model)

    audio_model=audio_model.to(device)
    
    audio_model.eval()
    all_predictions=[]
    all_targets=[]
    all_loss=[]
    with torch.no_grad():
        for audio_input, label in tqdm(val_loader):
            audio_input=audio_input.to(device)
            N=audio_input.size(0)

            # compute output
            audio_output=audio_model(audio_input)
            audio_output=torch.sigmoid(audio_output)
            predictions=audio_output.to("cpu").detach()

            all_predictions.append(predictions)
            all_targets.append(label)

            # compute the loss
            label=label.to(device)
            if isinstance(args.loss_fn,nn.CrossEntropyLoss):
                loss=args.loss_fn(audio_output, torch.argmax(label.long(), dim=1))
            else:
                loss=args.loss_fn(audio_output, label)

            valid_loss_meter.update(loss.item(), N)
        audio_output=torch.cat(all_predictions)
        target=torch.cat(all_targets)
        stats=calculate_stats(audio_output, target)

        # save the predictions here 
        # why should save the predictions here?
        
        exp_dir=args.exp_dir
        os.makedirs("{}/predictions".format(exp_dir),exist_ok=True)
        os.makedirs("{}/models".format(exp_dir),exist_ok=True)

        np.savetxt(exp_dir+"/predictions/target.csv", target, fmt="%.4f",delimiter=",")
        np.savetxt(exp_dir+"/predictions/predictions_{}.csv".format(epoch), audio_output, fmt="%.4f",delimiter=",")
        valid_loss=valid_loss_meter.avg
        valid_loss_meter.reset()
            
    return stats, valid_loss

def get_test_labels(audio_model, test_loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model,nn.DataParallel):
        audio_model=nn.DataParallel(audio_model)   # get the model from the dataparallel
    audio_model=audio_model.to(device)
    audio_model.eval()
    csv_label=args.label_csv
    save_wav_label_json=args.new_te_data
    wav_label_list=[]
    with torch.no_grad():
        for audio_input, wav_files  in tqdm(test_loader):
            audio_input=audio_input.to(device)
            # compute output
            audio_output=audio_model(audio_input)
            audio_output=torch.sigmoid(audio_output)
            audio_output=audio_output.to("cpu").detach().numpy()
            # write the predictions to csv file
            class_labels_indices=pd.read_csv(csv_label)
            for i, wav in enumerate(wav_files["wav"]):
                displays_names=[]
                for j, prob in enumerate(audio_output[i]):
                    if prob>=0.5:
                        displays_names.append(class_labels_indices.iloc[j,2])
                index1=np.argmax(audio_output[i])
                wav_label_list.append({"wav":wav,"label":class_labels_indices.iloc[index1,1],"display_name":class_labels_indices.iloc[index1,2],"display_all_names":displays_names})
    
    with open(save_wav_label_json, "w") as f:
        json.dump({"data":wav_label_list}, f,indent=4)
        print("results saved to {}".format(save_wav_label_json))

        



       



def validate_ensemble(args, epoch):
    # what is the purpose of this cum_predictions?
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

def validate_wa(audio_model,val_loader,args,start_epoch,end_epoch):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir
    
    sdA= torch.load(exp_dir+"/models/audio_model_"+str(start_epoch)+".pth",map_location=device)

    model_cnt=1
    for epoch in range(start_epoch+1, end_epoch+1):
        # what is this mean?
        sdB=torch.load(exp_dir+ "/models/audio_model_"+str(epoch)+".pth",map_location=device)
        
        # what is the purpose of this?
        for key in sdA:
            sdA[key]=sdA[key]+sdB[key]
        model_cnt+=1

        # 
        if args.save_model==False:
            os.remove(exp_dir+"/models/audio_model_"+str(epoch)+".pth")
    
    # average the model
    for key in sdA:
        sdA[key]=sdA[key]/float(model_cnt)
    
    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir+"/models/audio_model_wa.pth")

    stats, _=validate(audio_model,val_loader,args,"wa")
    return stats 



