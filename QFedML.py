from pathlib import Path
import tensorflow as tf
import warnings
import numpy as np
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded, info
from genomic_benchmarks.models.tf import vectorize_layer
from genomic_benchmarks.models.tf import get_basic_cnn_model_v0 as get_model
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy as np
import time
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from functools import partial
from qiskit_aer import AerSimulator
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.primitives import Sampler
import argparse
import os
import warnings
import sys
import warnings

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Create the parser
parser = argparse.ArgumentParser(description='Quantum Machine Learning Script')

# Add arguments
parser.add_argument('--num_clients', type=int, default=2, help='Number of clients')
parser.add_argument('--num_epochs', type=int, default=17, help='Number of epochs')
parser.add_argument('--max_train_iterations', type=int, default=100, help='Maximum training iterations')
parser.add_argument('--samples_per_epoch', type=int, default=2000, help='Samples per epoch')
parser.add_argument('--ansatz_reps', type=int, default=4, help='Number of ansatz repetitions')


## download the dataset

VERSION = 0
BATCH_SIZE = 64
DATASET = "demo_coding_vs_intergenomic_seqs"
if not is_downloaded(DATASET):
    download_dataset(DATASET)

info(DATASET)

SEQ_PATH = Path.home() / '.genomic_benchmarks' / DATASET
CLASSES = [x.stem for x in (SEQ_PATH/'train').iterdir() if x.is_dir()]
NUM_CLASSES = len(CLASSES)

train_dset = tf.keras.preprocessing.text_dataset_from_directory(
    SEQ_PATH / 'train',
    batch_size=BATCH_SIZE,
    class_names=CLASSES)


if NUM_CLASSES > 2:
    print("NUM_CLASESS > 2")
    train_dset = train_dset.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))

vectorize_layer.adapt(train_dset.map(lambda x, y: x))
VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
vectorize_layer.get_vocabulary()

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)-2, label

train_ds = train_dset.map(vectorize_text)

# Create the np_data_set list
np_data_set = []

for text_list, label_list in train_ds:
    for text, label in zip(text_list.numpy(), label_list.numpy()):
        sequence_dict = {"sequence": text.tolist(), "label": label.tolist()}
        np_data_set.append(sequence_dict)

# Convert the list of dictionaries to a NumPy array
np_data_set = np.array(np_data_set)

# Create the np_data_set
np_data_set = []

for text_list, label_list in train_ds:
    for text, label in zip(text_list.numpy(), label_list.numpy()):
        # Add an array of [-1, -1] to the sequence array
        sequence_with_padding = np.concatenate([text, np.full((256 - len(text)), -1)], axis=0)
        sequence_dict = {"sequence": sequence_with_padding.tolist(), "label": label.tolist()}
        np_data_set.append(sequence_dict)

# Convert the list of dictionaries to a NumPy array
np_data_set = np.array(np_data_set)


len(np_data_set[0]['sequence'])

# Shuffle the data set for testing
np.random.shuffle(np_data_set)

print("Print the first 5 examples in np_data_set")
for i, example in enumerate(np_data_set[:5]):
    print(f"Example {i + 1} - Sequence: {example['sequence']}, Label: {example['label']}")

print("\n\n")

# Split the data set into training and testing sets
np_train_data = np_data_set[:70000]
np_test_data = np_data_set[-5000:]

print(f"Length of np_train_data: {len(np_train_data)}")
print(f"Length of np_test_data: {len(np_test_data)}")
print("\n\n")

test_sequences = [data_point["sequence"] for data_point in np_test_data]
test_labels = [data_point["label"] for data_point in np_test_data]
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)


# Parse the arguments
args = parser.parse_args()

# Use the arguments
num_clients = args.num_clients
num_epochs = args.num_epochs
max_train_iterations = args.max_train_iterations
samples_per_epoch = args.samples_per_epoch
ansatz_reps = args.ansatz_reps

backend = AerSimulator()
fl_avg_weight_range = [0.1, 1]

# Create a list of clients with seperate data
class Client:
    def __init__(self, data):
        self.models = []
        self.primary_model = None
        self.data = data
        self.test_scores = []
        self.train_scores = []
        self.loss = []

def split_dataset(num_clients, num_epochs, samples_per_epoch):
  clients = []
  for i in range(num_clients):
    client_data = []
    for j in range(num_epochs):
      start_idx = (i*num_epochs*samples_per_epoch)+(j*samples_per_epoch)
      end_idx = (i*num_epochs*samples_per_epoch)+((j+1)*samples_per_epoch)
      client_data.append(np_train_data[start_idx:end_idx])
    clients.append(Client(client_data))
  return clients

clients = split_dataset(num_clients, num_epochs, samples_per_epoch)

clients[1].data[0][:3]


# Call back function: to be used in VQC training
itr = 0
def training_callback(weights, obj_func_eval):
        global itr
        itr += 1
        print(f"{itr} {obj_func_eval}", end=' | ')

# Function to get the accuracy of the model
def getAccuracy(weights, test_num = 200):
        num_features = len(test_sequences[0])
        n = (int)(np.log2(num_features))
        feature_map = RawFeatureVector(feature_dimension=num_features)
        ansatz = RealAmplitudes(num_qubits = n, reps=ansatz_reps)
        optimizer = COBYLA(maxiter=0)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=Sampler(),
            initial_point = weights
        )
        vqc.fit(test_sequences[:25], test_labels[:25])
        return vqc.score(test_sequences[:test_num], test_labels[:test_num])

# Function to create a model with the given weights
def create_model(weights = None):
  if(weights != None):
    num_features = len(test_sequences[0])
    n = (int)(np.log2(num_features))
    feature_map = RawFeatureVector(feature_dimension=num_features)
    ansatz = RealAmplitudes(num_qubits = n, reps=ansatz_reps)
    optimizer = COBYLA(maxiter=max_train_iterations)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=Sampler(),
        warm_start = True,
        initial_point  = weights,
        callback=partial(training_callback)
    )
    return vqc
  else:
    num_features = len(test_sequences[0])
    n = (int)(np.log2(num_features))
    feature_map = RawFeatureVector(feature_dimension=num_features)
    ansatz = RealAmplitudes(num_qubits = n, reps=ansatz_reps)
    optimizer = COBYLA(maxiter=max_train_iterations)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=Sampler(),
        warm_start = True,
        callback=partial(training_callback)
    )
    return vqc

def create_new_client_model(primary_model, current_epoch, global_model_weights):
  assigned_weights = [1/(current_epoch+2), (current_epoch+1)/(current_epoch+2)]
  primary_model_weights = primary_model.weights
  new_client_weights = []
  for index, _ in enumerate(primary_model_weights):
    new_client_weights.append(assigned_weights[1]*primary_model_weights[index] + assigned_weights[0]*global_model_weights[index])
  return create_model(new_client_weights)


# Temporary code to suppress all FutureWarnings for a cleaner output
warnings.simplefilter("ignore", FutureWarning)


def sort_epoch_results(epoch_results):
    pairs = zip(epoch_results['weights'], epoch_results['test_scores'])
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    sorted_weights, sorted_test_scores = zip(*sorted_pairs)
    sorted_epoch_results = {
        'weights': list(sorted_weights),
        'test_scores': list(sorted_test_scores)
    }
    return sorted_epoch_results

fl_avg_weight_range = [0.1, 1]

def scale_test_scores(sorted_epoch_results):
    min_test_score = sorted_epoch_results['test_scores'][0]
    max_test_score = sorted_epoch_results['test_scores'][-1]
    min_weight, max_weight = fl_avg_weight_range
    scaled_weights = [
        min_weight + (max_weight - min_weight) * (test_score - min_test_score) / (max_test_score - min_test_score)
        for test_score in sorted_epoch_results['test_scores']
    ]
    sorted_epoch_results['fl_avg_weights'] = scaled_weights
    return sorted_epoch_results

def calculate_weighted_average(model_weights, fl_avg_weights):
    weighted_sum_weights = []
    for index in range(len(model_weights[0])):
      weighted_sum_weights.append(0)
      weighted_sum_weights[index] = sum([(weights_array[index]* avg_weight) for weights_array, avg_weight  in zip(model_weights, fl_avg_weights)])/sum(fl_avg_weights)
    return weighted_sum_weights

def weighted_average(epoch_results, global_model_weights_last_epoch = None, global_model_accuracy_last_epoch = None):
  if(global_model_weights_last_epoch != None):
    epoch_results['weights'].append(global_model_weights_last_epoch)
    epoch_results['test_scores'].append(global_model_accuracy_last_epoch)

  if all(epoch_results['test_scores'][0] == x for x in epoch_results['test_scores']):
      print("Equal test scores received")
      return simple_averaging(epoch_results)
  epoch_results = sort_epoch_results(epoch_results)
  epoch_results = scale_test_scores(epoch_results)
  weighted_average_weights_curr_epoch = calculate_weighted_average(epoch_results['weights'], epoch_results['fl_avg_weights'])
  return weighted_average_weights_curr_epoch



def weighted_average_best_pick(epoch_results, global_model_weights_last_epoch = None, global_model_accuracy_last_epoch = None, best_pick_cutoff = 0.5):
  if(global_model_weights_last_epoch != None):
    epoch_results['weights'].append(global_model_weights_last_epoch)
    epoch_results['test_scores'].append(global_model_accuracy_last_epoch)
  if all(epoch_results['test_scores'][0] == x for x in epoch_results['test_scores']):
      print("Equal test scores received")
      return simple_averaging(epoch_results)
  epoch_results = sort_epoch_results(epoch_results)
  epoch_results = scale_test_scores(epoch_results)
  new_weights = []
  new_test_scores = []
  new_fl_avg_weights = []

  for index, fl_avg_weight in enumerate(epoch_results['fl_avg_weights']):
      if fl_avg_weight >= best_pick_cutoff:
          new_weights.append(epoch_results['weights'][index])
          new_test_scores.append(epoch_results['test_scores'][index])
          new_fl_avg_weights.append(fl_avg_weight)

  epoch_results['weights'] = new_weights
  epoch_results['test_scores'] = new_test_scores
  epoch_results['fl_avg_weights'] = new_fl_avg_weights
  weighted_average_weights_curr_epoch = calculate_weighted_average(epoch_results['weights'], epoch_results['fl_avg_weights'])
  return weighted_average_weights_curr_epoch

def simple_averaging(epoch_results, global_model_weights_last_epoch = None, global_model_accuracy_last_epoch = None):
  if(global_model_weights_last_epoch != None):
    epoch_results['weights'].append(global_model_weights_last_epoch)
    epoch_results['test_scores'].append(global_model_accuracy_last_epoch)

  epoch_weights = epoch_results['weights']
  averages = []
  for col in range(len(epoch_weights[0])):
      col_sum = 0
      for row in range(len(epoch_weights)):
          col_sum += epoch_weights[row][col]
      col_avg = col_sum / len(epoch_weights)
      averages.append(col_avg)

  return averages


def train(data, model = None):
  if model is None:
    model = create_model()

  train_sequences = [data_point["sequence"] for data_point in data]
  train_labels = [data_point["label"] for data_point in data]

  train_sequences = np.array(train_sequences)
  train_labels = np.array(train_labels)

  print("Train Sequences Shape:", train_sequences.shape)
  print("Train Labels Shape:", train_labels.shape)

  print("Training Started")
  start_time = time.time()
  model.fit(train_sequences, train_labels)
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"\nTraining complete. Time taken: {elapsed_time} seconds.")

  print(f"SCORING MODEL")
  train_score_q = model.score(train_sequences, train_labels)
  test_score_q = model.score(test_sequences[:200], test_labels[:200])
  return train_score_q, test_score_q, model

fl_techniques = {
    'Best Pick': weighted_average_best_pick,
    'Weighted Averaging': weighted_average,
    'Averaging': simple_averaging
}

clients_2d_array = [[] for _ in range(len(fl_techniques))]

for index, (technique_name, _) in enumerate(fl_techniques.items()):
        for client in clients:
          client_copy = Client(client.data)
          clients_2d_array[index].append(client_copy)

clients_2d_array

global_model_weights = []
global_model_accuracy = []

# Train the models for each client
for outer_idx, clients in enumerate(clients_2d_array):
  technique_name = list(fl_techniques.keys())[outer_idx]
  technique_function = list(fl_techniques.values())[outer_idx]
  print(f"Technique Name: {technique_name}")
  global_model_weights.append([])
  global_model_accuracy.append([])
  for epoch in range(num_epochs):
    epoch_results = {
        'weights': [],
        'test_scores': []
    }
    print(f"epoch: {epoch}")

    for index, client in enumerate(clients):
      print(f"Index: {index}, Client: {client}")

      if client.primary_model is None:
        train_score_q, test_score_q, model = train(data = client.data[epoch])
        client.models.append(model)
        client.test_scores.append(test_score_q)
        client.train_scores.append(train_score_q)
        client.loss.append(model.fit_result.fun)
        # Print the values
        print("Train Score:", train_score_q)
        print("Test Score:", test_score_q)
        print("\n\n")
        epoch_results['weights'].append(model.weights)
        epoch_results['test_scores'].append(test_score_q)

      else:
        train_score_q, test_score_q, model = train(data = client.data[epoch], model = client.primary_model)
        client.models.append(model)
        client.test_scores.append(test_score_q)
        client.train_scores.append(train_score_q)
        client.loss.append(model.fit_result.fun)
        print("Train Score:", train_score_q)
        print("Test Score:", test_score_q)
        print("\n\n")
        epoch_results['weights'].append(model.weights)
        epoch_results['test_scores'].append(test_score_q)

    # Client training complete accumulate results for global model
    new_global_weights = []
    if(epoch == 0):
      new_global_weights = technique_function(epoch_results)
    else:
      new_global_weights = technique_function(epoch_results, global_model_weights[outer_idx][epoch - 1], global_model_accuracy[outer_idx][epoch - 1])
    global_model_weights[outer_idx].append(new_global_weights)
    new_model_with_global_weights = create_model(weights = global_model_weights[outer_idx][epoch])

    for index, client in enumerate(clients):
      client.primary_model = create_new_client_model(client.models[-1], epoch, global_model_weights[outer_idx][epoch])

    global_accuracy = getAccuracy(global_model_weights[outer_idx][epoch], len(test_sequences[:200]))
    global_model_accuracy[outer_idx].append(global_accuracy)
    print(f"Technique Name: {technique_name}")
    print(f"Global Model Accuracy In Epoch {epoch}: {global_accuracy}")
    print("----------------------------------------------------------")
    print("\n\n")

# Training and testing complete: save the graphs
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

print("Script directory:", script_dir)
print("\n")

image_path = f"{script_dir}/graphs_{num_clients}_{num_epochs}_{max_train_iterations}_{samples_per_epoch}_{ansatz_reps}" 

if not os.path.exists(image_path):
    os.makedirs(image_path)

# Create two figures, one for train scores and one for test scores

for idx, clients in enumerate(clients_2d_array):

  technique_name = list(fl_techniques.keys())[idx]

  plt.figure(figsize=(8, 6))

  ax1 = plt.gca()
  ax2 = ax1.twinx()

  for client in clients:
      client_index = clients.index(client) + 1
      ax1.plot(client.train_scores, label=f'Client {client_index} Train Score')
      ax2.plot(client.loss, label=f'Client {client_index} Loss', linestyle='--')

  ax1.set_xlabel('Epochs', fontsize=14) 
  ax1.set_ylabel('Train Score', color='blue', fontsize=14)  
  ax2.set_ylabel('Loss', color='red', fontsize=14)  
  ax1.set_title(f"Train Scores and Losses for All Clients - {technique_name}", fontsize=16)  
  lines_1, labels_1 = ax1.get_legend_handles_labels()
  lines_2, labels_2 = ax2.get_legend_handles_labels()
  ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12) 

  plt.savefig(f'{image_path}/{technique_name}_Train_Loss.png')



  plt.figure(figsize=(8, 6))

  for client in clients:
      plt.plot(client.test_scores, label=f'Client {clients.index(client) + 1}')

  plt.xlabel('Epochs', fontsize=14)  
  plt.ylabel('Test Score',fontsize=14)  
  plt.title(f"Top-1 Accuracy for All Clients - {technique_name}", fontsize=16)  
  plt.legend(fontsize=14) 

  plt.savefig(f'{image_path}/{technique_name}_Top1_Clients.png')

for idx, clients in enumerate(clients_2d_array):

  technique_name = list(fl_techniques.keys())[idx]
  plt.figure(figsize=(8, 6))

  for client in clients:
      plt.plot(client.test_scores, label=f'Client {clients.index(client) + 1}')

  plt.plot(global_model_accuracy[idx], label='Global Model Accuracy', linestyle='--', color='black')

  plt.xlabel('Epochs', fontsize=14) 
  plt.ylabel('Scores', fontsize=14) 
  plt.title(f"Top-1 Accuracy - {technique_name}", fontsize=16)  
  plt.legend(fontsize=12) 

  plt.savefig(f'{image_path}/{technique_name}__Top1_Clients_Global.png')

np_final_test_data = np_data_set[70000:]

final_test_sequences = [data_point["sequence"] for data_point in np_final_test_data]
final_test_labels = [data_point["label"] for data_point in np_final_test_data]
final_test_sequences = np.array(final_test_sequences)
final_test_labels = np.array(final_test_labels)

def getFinalAccuracy(weights):
        num_features = len(test_sequences[0])
        n = (int)(np.log2(num_features))
        feature_map = RawFeatureVector(feature_dimension=num_features)
        ansatz = RealAmplitudes(num_qubits = n, reps=ansatz_reps)
        optimizer = COBYLA(maxiter=0)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point = weights,
            sampler=Sampler()
        )
        vqc.fit(test_sequences[:25], test_labels[:25])
        return vqc.score(final_test_sequences, final_test_labels)

final_results = []
for idx, row in enumerate(global_model_weights):
  final_results.append([])
  for global_model_weight in row:
    final_results[idx].append(getFinalAccuracy(global_model_weight))

print("Final Results ")
print(final_results)



final_results = np.array(final_results)  

plt.figure(figsize=(10, 8))

for row_index in range(final_results.shape[0]):
    data_row = final_results[row_index]
    technique_name = list(fl_techniques.keys())[row_index]
    plt.plot(data_row, label=f'Global Model Accuracy For: {technique_name}')

plt.xlabel('Epochs', fontsize=14)  
plt.ylabel('Value', fontsize=14)  
plt.title('Global Model Accuracy', fontsize=16)  
plt.legend(fontsize=12)

plt.xticks(range(final_results.shape[1]))

plt.savefig(f'{image_path}/Final_global_accuracy.png')


print("Saving models")

for idx, clients in enumerate(clients_2d_array):
  technique_name = list(fl_techniques.keys())[idx]
  technique_path = f"{script_dir}/{technique_name}_models" 
  if not os.path.exists(technique_path):
    os.makedirs(technique_path)
  for client in clients:
    client_index = clients.index(client) + 1
    client_path = f"{technique_path}/client{client_index}"
    if not os.path.exists(client_path):
      os.makedirs(client_path)
    for model_index, vqc_model in enumerate(client.models):
       vqc_model.save(f"{client_path}/model_{model_index}")


    