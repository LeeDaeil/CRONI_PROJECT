from Old_Ver import Network
import numpy as np
import pickle


class trainner:
    def __init__(self):
        pass
    def load_db(self, file):
        with open(file, 'rb') as f:
            db = pickle.load(f)'' \
                               '' \
                               '' \
                               '' \
                               ''

        test_db = ['ab15-07_1004.pkl', 'ab15-07_1006.pkl']

        self.train_x_db = [db[_]['train_x_db'] for _ in db.keys() if not _ in test_db]
        self.train_y_db = [db[_]['train_y_db'] for _ in db.keys() if not _ in test_db]

        self.test_x_db = [db[_]['train_x_db'] for _ in db.keys() if _ in test_db]
        self.test_y_db = [db[_]['train_y_db'] for _ in db.keys() if _ in test_db]

        # make_network
        initial_key = list(db.keys())[0]
        _, self.time_leg, self.input_pa = np.shape(db[initial_key]['train_x_db'])
        _, self.output_pa = np.shape(db[initial_key]['train_y_db'])

    def train_control(self, type_net):
        self.main_train(type_net, False)

    def train_clasiffer(self, type_net):
        self.main_train(type_net, True)

    def main_train(self, type_net, classify):
        network = Network.MainNet(net_type=type_net, input_pa=self.input_pa, output_pa=self.output_pa,
                                  time_leg=self.time_leg, classify=classify)
        history = network.main_network.fit(np.concatenate(tuple(self.train_x_db)), np.concatenate(tuple(self.train_y_db)),
                                           batch_size=50,
                                           epochs=100, verbose=0, shuffle=True, validation_split=0.1)

        print(f'Type: {type_net}\nACC:{history.history["acc"][-1]}, {history.history["val_acc"][-1]}'
              f'\nLoss: {history.history["loss"][-1]}, {history.history["val_loss"][-1]}')

        with open(f'{type_net}_save_history.bin', 'bw') as f:
            pickle.dump(history, f)

    def show_network(self, type_net, classify):
        network = Network.MainNet(net_type=type_net, input_pa=self.input_pa, output_pa=self.output_pa,
                                  time_leg=self.time_leg, classify=classify)