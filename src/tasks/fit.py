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