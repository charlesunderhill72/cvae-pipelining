"""This will be the module where I do this procedure:
        batch_X_c = X_c_shuffled_padded[iter:iter+batch_size]
        batch_X = X_shuffled_padded[iter:iter+batch_size]
        #print(batch_X.size())
        optimizer.zero_grad()
        outputs = model(batch_X_c.to(device).float())
        reconstruction, mu, log_var = outputs
        #print(reconstruction.size())
        bce_loss = criterion(reconstruction, batch_X.to(device).float())
        loss = final_loss(bce_loss, mu, log_var)
        loss.backward()
        
   Which is essentially fitting the model. This might be considered a task. 
   I will need to set the inputs to be the corrupted data inputs, which I can 
   do per batch, and the labels are the uncorrupted inputs. The corruption will be 
   part of the preprocessing stage. 
"""
"""
for epoch_idx in range(num_epochs):
        losses = []
        for i, sample in tqdm(enumerate(training_generator)):
            optimizer.zero_grad()
            im = im.float().to(device)
            im = pad_to_power_of_2(im)

            recon, mu, log_var = model(im)
            mse_loss = criterion(recon, im)
            loss = final_loss(mse_loss, mu, log_var)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch_idx + 1,
                np.mean(losses),
            ))

        torch.save(model.state_dict(), os.path.join(autoencoder_config['task_name'],
                                                autoencoder_config['ckpt_name']))
"""

import torch

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, sample_in, sample_out, criterion, losses, loss):
    return