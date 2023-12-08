# Import the W&B Python Library and log into W&B
import wandb
import preprocessing
import utils
import autoencoder
import custom_dataset

wandb.login()


# 1: Define objective/training function
def objective(config):
    k, learning_rate, num_epochs, batch_size = config["k"], config["learning_rate"], config["num_epochs"], config["batch_size"]
    train_set, test_set, validation_set, train_df, genre_df = preprocessing.get_data()
    train_set = custom_dataset.CustomDataset(train_set)
    validation_set = custom_dataset.CustomDataset(validation_set)
    loss = autoencoder.train_autoencoder(train_set, validation_set, wandb=wandb, k=k, learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size)
    return loss


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "k": {"distribution": "int_uniform", "min": 100, "max": 500},
        "learning_rate": {"distribution": "uniform", "min": 0.00001, "max": 0.0001},
        "weight_decay": {"distribution": "uniform", "min": 0.000001, "max": 0.00001},
        # num_epochs stay at 100 for now
        "num_epochs": {"values": [100]},
        "batch_size": {"values": [100]}
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)