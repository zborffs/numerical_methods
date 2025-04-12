import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# Generate training data (7x7 grid)
n = 9
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xv, yv = np.meshgrid(x, y)
xv_flat = xv.flatten()
yv_flat = yv.flatten()

# Strip boundaries
mask = (xv_flat > 0) & (xv_flat < 1) & (yv_flat > 0) & (yv_flat < 1)
x_train = xv_flat[mask]
y_train = yv_flat[mask]

x_tensor = torch.tensor(x_train, requires_grad=True)
y_tensor = torch.tensor(y_train, requires_grad=True)


# Neural network
class NN(torch.nn.Module):
    def __init__(self, hidden=10):
        super().__init__()
        self.l1 = torch.nn.Linear(2, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

    def forward(self, x, y):
        inputs = torch.stack([x, y], dim=1)
        z = torch.tanh(self.l1(inputs))
        return self.l2(z).squeeze()
    # def forward(self, xb):
    #     # inputs = torch.stack([x, y], dim=1)
    #     z = torch.tanh(self.l1(xb))
    #     return self.l2(z)


model = NN(10)

# Trial solution
def U(x, y, model):
    term1 = 2 * y * torch.sin(np.pi * x)
    B = x * (1 - x) * y * (1 - y)
    return term1 + B * model(x, y)


# Loss function
def loss_fn(model):
    u = U(x_tensor, y_tensor, model)
    u_x = torch.autograd.grad(u, x_tensor, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(u_x, x_tensor, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
    u_y = torch.autograd.grad(u, y_tensor, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_yy = torch.autograd.grad(u_y, y_tensor, create_graph=True, grad_outputs=torch.ones_like(u_y))[0]
    residual = u_xx + u_yy - (2 - (np.pi**2)*(y_tensor**2)) * torch.sin(np.pi * x_tensor)
    return 0.5 * torch.sum(residual**2)


# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_history = []

for epoch in range(2500):
    optimizer.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("done training")

# Plot loss
plt.semilogy(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot solution and error
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
with torch.no_grad():
    u_pred = U(torch.tensor(xx.ravel()), torch.tensor(yy.ravel()), model).numpy().reshape(xx.shape)
u_exact = (yy**2) * np.sin(np.pi * xx)
error = 1/2*u_pred - u_exact

print("Max error = ",np.max(error))
print("Mean error = ",np.mean(error))
print("RMS error = ",np.sqrt(np.mean(error**2)))

plt.figure()
plt.contourf(xx, yy, 1/2*u_pred, levels=20)
plt.colorbar()
plt.title('Estimated Solution')

plt.figure()
plt.contourf(xx, yy, u_exact, levels=20)
plt.colorbar()
plt.title('Exact Solution Solution')

plt.figure()
plt.contourf(xx, yy, error, levels=20)
plt.colorbar()
plt.title('Error')
plt.show()