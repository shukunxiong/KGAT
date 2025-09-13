import os

import torch
from tqdm import tqdm

from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
        #----------------------#
        #   Zero out gradients
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   Forward propagation
            #----------------------#
            # dbox, cls, origin_cls, anchors, strides 
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            #----------------------#
            #   Backward propagation
            #----------------------#
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   Forward propagation
                #----------------------#
                outputs         = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)

            #----------------------#
            #   Backward propagation
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            #----------------------#
            #   Zero out gradients
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   Forward propagation
            #----------------------#
            outputs     = model_train_eval(images)
            loss_value  = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   Save model weights
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        
def fit_one_epoch_Knowledge(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, adv,local_rank=0,n=4):
    loss        = 0
    val_loss    = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images_clean, bboxes_clean = batch
        
        with torch.no_grad():
            if cuda:
                images_clean = images_clean.cuda(local_rank)
                bboxes_clean = bboxes_clean.cuda(local_rank)
        #----------------------#
        #  Zero out gradients
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   Forward propagation
            #----------------------#
            # dbox, cls, origin_cls, anchors, strides
            # outputs = model_train(images_clean)
            images_for_generate_adv = images_clean[:n].clone().detach()
            images_adv = adv(images_for_generate_adv, bboxes_clean,model_train)
            images = torch.cat((images_adv, images_clean), dim=0).to(images_clean.device)
            outputs = model_train(images)
            # ----------Find indices of boxes corresponding to the first n images used for generating adversarial examples
            valid_indices = (bboxes_clean[:, 0] <= (n-1))
            bboxes_adv= bboxes_clean[valid_indices]
            bboxes = torch.cat((bboxes_clean, bboxes_adv), dim=0).to(bboxes_clean.device)
            loss_value = yolo_loss(images,outputs, bboxes)
            #----------------------#
            #   Backward propagation
            #----------------------#
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   Forward propagation
                #----------------------#
                # outputs         = model_train(images)
                images_for_generate_adv = images_clean[:n].clone().detach()
                images_adv = adv(images_for_generate_adv, bboxes_clean,model_train)
                images = torch.cat((images_adv, images_clean), dim=0).to(images_clean.device)
                outputs = model_train(images)
                # ----------Find indices of boxes corresponding to the first n images used for generating adversarial examples
                valid_indices = (bboxes_clean[:, 0] <= (n-1))
                bboxes_adv= bboxes_clean[valid_indices]
                bboxes = torch.cat((bboxes_clean, bboxes_adv), dim=0).to(bboxes_clean.device)
                loss_value = yolo_loss(images,outputs, bboxes)

            #----------------------#
            #   Backward propagation
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            #----------------------#
            #   Zero out gradients
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   Forward propagation
            #----------------------#
            outputs     = model_train_eval(images)
            loss_value  = yolo_loss(images,outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   Save model weights
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        
    loss        = 0
    val_loss    = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images_clean, bboxes_clean = batch
        
        with torch.no_grad():
            if cuda:
                images_clean = images_clean.cuda(local_rank)
                bboxes_clean = bboxes_clean.cuda(local_rank)
        #----------------------#
        #   Zero out gradients
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   Forward propagation
            #----------------------#
            # dbox, cls, origin_cls, anchors, strides
            # outputs = model_train(images_clean)
            images_for_generate_adv = images_clean[:n].clone().detach()
            images_adv = adv(images_for_generate_adv, bboxes_clean,model_train)
            images = torch.cat((images_adv, images_clean), dim=0).to(images_clean.device)
            outputs = model_train(images)
            # ----------Find indices of boxes corresponding to the first n images used for generating adversarial examples
            valid_indices = (bboxes_clean[:, 0] <= (n-1))
            bboxes_adv= bboxes_clean[valid_indices]
            bboxes = torch.cat((bboxes_clean, bboxes_adv), dim=0).to(bboxes_clean.device)
            loss_value = yolo_loss(images,outputs, bboxes)
            #----------------------#
            #   Backward propagation
            #----------------------#
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   Forward propagation
                #----------------------#
                # outputs         = model_train(images)
                images_for_generate_adv = images_clean[:n].clone().detach()
                images_adv = adv(images_for_generate_adv, bboxes_clean,model_train)
                images = torch.cat((images_adv, images_clean), dim=0).to(images_clean.device)
                outputs = model_train(images)
                # ----------Find indices of boxes corresponding to the first n images used for generating adversarial examples
                valid_indices = (bboxes_clean[:, 0] <= (n-1))
                bboxes_adv= bboxes_clean[valid_indices]
                bboxes = torch.cat((bboxes_clean, bboxes_adv), dim=0).to(bboxes_clean.device)
                loss_value = yolo_loss(images,outputs, bboxes)

            #----------------------#
            #   Backward propagation
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            #----------------------#
            #   Zero out gradients
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   Forward propagation
            #----------------------#
            outputs     = model_train_eval(images)
            loss_value  = yolo_loss(images,outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   Save model weights
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))