# include <iostream>
# include <cmath>

double sigmoid(double s) {
    return 1.0 / (1.0 + exp(-s));
}

double sigmoid_derivative(double s) {
    return s * (1.0 - s); // Da s = sigmoid(x), gilt: sigma'(x) = sigma(x) * (1 - sigma(x))
}

// Neuronenklasse
class Neuron {
    public:
        double *weights;  // Array für mehrere Gewichte
        int numInputs;
        double bias;
        double output;
        double delta;

        Neuron() : weights(nullptr), numInputs(0), bias(0.0), output(0.0), delta(0.0) {}

        void init(int numInputs_) {
            numInputs = numInputs_;
            if (weights) {
                delete[] weights;
            }
            if (numInputs > 0) {
                weights = new double[numInputs]; // Initialisiere Gewicht-Array
                for (int i = 0; i < numInputs; ++i) {
                    weights[i] = ((double) rand() / RAND_MAX) * 2 - 1; // Zufällige Initialisierung (-1 bis 1)
                }
            } else {
                weights = nullptr;
            }
            bias = ((double) rand() / RAND_MAX) * 2 - 1; // Zufälliger Bias
            output = 0.0;
        }

        ~Neuron() {
            delete[] weights; // Speicher freigeben
        }
    };

// Layerklasse
class Layer {
public:
    Neuron* neurons;
    int numNeurons;
    int numInputs;
    //double* inputs;

    Layer() {}

    void makeNeurons() {
        neurons = new Neuron[numNeurons]; // Neuronen erzeugen
        for (int i = 0; i < numNeurons; ++i) {
            neurons[i].init(numInputs); // Initialisierung mit numInputs
        }
    }

    ~Layer() {
        delete[] neurons; // Speicher freigeben
    }
};

class NeuronNet {
public:
    Layer* layers;
    int numLayers;
    NeuronNet(int* layer_struc, int maxLayer) {
        numLayers = maxLayer;
        layers = new Layer[numLayers];

        for (int i = 0; i < numLayers; ++i) {
            layers[i].numNeurons = layer_struc[i];
            if (i == 0) {
                layers[i].numInputs = 0;
            } else {
                layers[i].numInputs = layers[i - 1].numNeurons;
            }
            layers[i].makeNeurons();
        }
    }

    ~NeuronNet() {
        delete[] layers;
    }

    // Vorwärtsfunktion
    void forward(double *input) {
        // Setze die Eingabewerte
        for (int i = 0; i < layers[0].numNeurons; ++i) {
            layers[0].neurons[i].output = input[i];
        }

        // Berechne den Output für jede Schicht
        for (int i = 1; i < numLayers; ++i) {
            for (int j = 0; j < layers[i].numNeurons; ++j) {
                double sum = layers[i].neurons[j].bias; // Bias hinzufügen
                for (int k = 0; k < layers[i].numInputs; ++k) {
                    sum += layers[i - 1].neurons[k].output * layers[i].neurons[j].weights[k];
                }
                layers[i].neurons[j].output = sigmoid(sum);
            }
        }
    }

    // Backpropagation: Training mit einem Lernfaktor
    void train(double* input, double* target, double learning_rate) {
        forward(input);

        // Fehlerberechnung für den Output-Layer
        for (int i = 0; i < layers[numLayers - 1].numNeurons; ++i) {
            double error = target[i] - layers[numLayers - 1].neurons[i].output;
            layers[numLayers - 1].neurons[i].delta = error * sigmoid_derivative(layers[numLayers - 1].neurons[i].output);
        }

        // Fehler zurückpropagieren (Backpropagation)
        for (int i = numLayers - 2; i > 0; --i) { // Gehe rückwärts durch die Hidden-Layer
        for (int j = 0; j < layers[i].numNeurons; ++j) { // Alle Neuronen in dieser Schicht
        double error = 0.0;
        for (int k = 0; k < layers[i + 1].numNeurons; ++k) { // Alle Neuronen der nächsten Schicht
            error += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].weights[j];
            //Hier wird jetzt weights[j] verwendet, weil jedes Neuron mehrere Eingangsgewichte hat
        }
        layers[i].neurons[j].delta = error * sigmoid_derivative(layers[i].neurons[j].output);
    }
}


        // Gewichte und Biases aktualisieren
        for (int i = 1; i < numLayers; ++i) { // Alle Schichten außer Input-Layer
            for (int j = 0; j < layers[i].numNeurons; ++j) { // Alle Neuronen in der Schicht
                for (int k = 0; k < layers[i].numInputs; ++k) { // Alle Eingänge des Neurons
                // Richtige Gewichtsanpassung
                layers[i].neurons[j].weights[k] += learning_rate * layers[i].neurons[j].delta * layers[i - 1].neurons[k].output;
                }
                // Bias ebenfalls anpassen
                layers[i].neurons[j].bias += learning_rate * layers[i].neurons[j].delta;
    }
}
    }
};

// Hauptprogramm
int main() {
    //Struktur des Netzes Festlegen Anzahl der Layer mit Anz. Neuronen
    int layer_struc[] = {2, 3, 1};
    NeuronNet net(layer_struc, sizeof(layer_struc) / sizeof(int));

    // Trainingsdaten für UND-Funktion
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4][1] = {
        {0.0},
        {0.0},
        {0.0},
        {1.0}
    };

    double learning_rate = 0.5;
    int epochs = 5000;

    // Training des Netzwerks
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        for (int i = 0; i < 4; ++i) {
            //Uebergabe der Zeilenzeiger von inputs und targets
            net.train(inputs[i], targets[i], learning_rate);

            // Fehler berechnen
            net.forward(inputs[i]);
            double error = pow(targets[i][0] - net.layers[sizeof(layer_struc)/sizeof(int)-1].neurons[0].output, 2);
            total_error += error;
        }
        // Nach 500 Iterationen Zwischenergebnis des Fehlers ausgeben
        if (epoch % 500 == 0) {
            std::cout << "Epoche " << epoch << " - Fehler: " << total_error / 4.0 << std::endl;
        }
    }

    // Netz testen
    std::cout << "\nTrainiertes Netz:\n";
    for (int i = 0; i < 4; ++i) {
        net.forward(inputs[i]);
        std::cout << "Eingang: " << inputs[i][0] << ", " << inputs[i][1]
                  << " -> Ausgabe: " << net.layers[sizeof(layer_struc)/sizeof(int)-1].neurons[0].output << std::endl;
    }

    return 0;
}
