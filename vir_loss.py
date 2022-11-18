import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable









def vir_reweight(logits, logits_adv, target,alpha =7.0, bet=0.007, pw = 10.0, method = 'at'):
    
    kl = nn.KLDivLoss(reduction='none')
    
    adv_probs = F.softmax(logits_adv, dim=1)
    
    nat_probs = F.softmax(logits, dim=1)
    
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    
    divg = torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim =1) 
    
    if method != 'at':
        pw = 3.0
        bet = 1.7
        alpha = 8.0
    
    
    reweight =  float(alpha) * torch.exp(-pw * true_probs)* divg + bet 
    
    reweight = reweight.detach()
     
   
    return  reweight
    


def vir_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=5.0,
              distance='l_inf',
              method = 'at'):
    kl = nn.KLDivLoss(reduction='none')
    
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()   #change to eval as default
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                #logt_nat = model(x_natural)
                logt = model(x_adv)
                if method != 'trades':
                    #if mode != 'margin':
                    loss_ce = F.cross_entropy(logt, y)
                       
                    
                  
                  #loss_ce = loss_ce.mean()         #F.cross_entropy(logt, y)
                else:
                     
                    loss_ce  =  criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                          F.softmax(model(x_natural), dim=1)) 
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
     
    logits = model(x_natural)

    logits_adv = model(x_adv)
  
    
    
    loss_adv = F.cross_entropy(logits_adv, y, reduction= 'none')
    loss_nat = F.cross_entropy(logits, y, reduction = 'none')

     
    loss_ce = F.cross_entropy(logits, y, reduction = 'none')
    
   
     
    if method == 'at':
        
        rw = vir_reweight(logits, logits_adv, y,method = 'at')
             
        loss_wei =  (loss_adv*rw).mean()
    elif method == 'mart':
        rw = vir_loss(model, x_natural, y, method = 'mail')
        logits = model(x_natural)
        logits_adv = model(x_adv)
      

        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, y, reduction = 'none') + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction = 'none')
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss_wei = (loss_adv * rw).mean() + float(beta) * loss_robust      
               
    else:
        rw = vir_reweight(logits, logits_adv, y,method = 'trades')
        loss_natural = F.cross_entropy(logits, y, reduction = 'none')
      
        loss_robust = torch.sum(kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)), dim=1) 
        loss_wei = loss_natural + float(beta) * loss_robust*rw 
        loss_wei = loss_wei.mean()


    return loss_wei

