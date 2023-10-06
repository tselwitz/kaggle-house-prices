import torch
import torch.nn.utils as utils

class Trainer():
  def __init__(self, epochs, model, loss_fn, optimizer):
    self.epochs = epochs
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.loss_old = None
    self.test_loss_old = None
    
  # Calculate accuracy (a classification metric)
  def accuracy_fn(y_true, y_pred):
      """Calculates accuracy between truth labels and predictions.

      Args:
          y_true (torch.Tensor): Truth labels for predictions.
          y_pred (torch.Tensor): Predictions to be compared to predictions.

      Returns:
          [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
      """
      correct = torch.eq(y_true, y_pred).sum().item()
      acc = (correct / len(y_pred)) * 100
      return acc
  
  def train_loop(self, y_train, X_train, y_test, X_test):
    self.model.train()
    y_logits = self.model(X_train).squeeze()
    self.loss = self.loss_fn(input=y_logits, target=y_train)
    self.optimizer.zero_grad()
    self.loss.backward()
    utils.clip_grad_norm_(self.model.parameters(), .8)
    self.optimizer.step()
    self.model.eval()
    with torch.inference_mode():
      test_logits = self.model(X_test).squeeze()
      self.test_loss = self.loss_fn(test_logits, y_test)
    if self.loss_old and self.test_loss_old:
      if abs(self.loss + self.test_loss) >= abs(self.loss_old + self.test_loss_old):
        return 0
    self.test_loss_old = self.test_loss
    self.loss_old = self.loss
    return 1
  
  def show_test(self, epoch):
      print(f"Epoch {epoch}| Loss: {self.loss:.5E} | Test Loss: {self.test_loss:.5E}") 
  
  def train(self, y_train, X_train, y_test, X_test, test_cadence):
    for e in range(self.epochs):
      status = self.train_loop(y_train, X_train, y_test, X_test)
      if e % test_cadence == 0:
        self.show_test(e)
      if status == 0:
        self.show_test(e)
        return