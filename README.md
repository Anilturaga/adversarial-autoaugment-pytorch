
# Untangle-Adversarial-Autoaugment

Pytorch Implementation Of [AdversarialAutoAugment(ICLR2020)](https://arxiv.org/pdf/1912.11188.pdf) as a feature for Untangle vision platform (v 2.0.0)

> Refer to example_usage.py for API usage.

## Rules

- Use `if __name__ == '__main__'`in client file to call distributed data parallel

- Transforms should have compose. If no transforms, use [] in compose

- No need to use ToTensor in transforms as we add augmentation transforms and then a ToTensor transform internally

  

## File structure

- Trained model is automatically returned with UntangleAI.train_augment(...)

- All files are saved in mname/train_augment/experiment_ID/

- models folder has several model checkpoints for each epoch and top state_dicts

- logs.pkl have train and test logs and top policies

## Controller

    (module): Controller(
    
    (embedding): Embedding(25, 32)
    
    (lstm): LSTMCell(32, 100)
    
    (outop): Linear(in_features=100, out_features=15, bias=True)
    
    (outmag): Linear(in_features=100, out_features=10, bias=True)
    
    )

Implemented from [SeongwoongJo's repo](https://github.com/SeongwoongJo/adversarial-autoaugment-pytorch)