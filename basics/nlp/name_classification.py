import torch
import torch.nn as nn
import random

from utils import N_LETTERS, load_data, line_to_tensor


def random_training_example(category_lines, all_categories):
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def category_from_output(output, all_categories):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


def train(model, criterion, optimizer, line_tensor, category_tensor):
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


def predict(rnn, input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
        print(guess)


def main():
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)

    n_hidden = 128
    rnn = RNN(N_LETTERS, n_hidden, n_categories)

    criterion = nn.NLLLoss()
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    n_iters = 100000
    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example(
            category_lines, all_categories
        )

        output, loss = train(rnn, criterion, optimizer, line_tensor, category_tensor)
        current_loss += loss

        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        if (i + 1) % print_steps == 0:
            guess = category_from_output(output, all_categories)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")


if __name__ == "__main__":
    main()
