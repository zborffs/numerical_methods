J = 10;
kx_vec = [pi, pi, (J-1)*pi];
ky_vec = [pi, 2*pi, (J-1)*pi];
h = 1 / (J);
lambda_vec = [];
v_vec = cell(3,1);

for ii = 1:3
    kx = kx_vec(ii);
    ky = ky_vec(ii);
    lambda_vec(ii) = 2 / h^2 * (cos(kx * h) + cos(ky * h) - 2);
    v_vec{ii} = @(x,y)sin(kx * x) .* sin(ky * y);
end


t = linspace(0, 1, 1000);
[x,y] = meshgrid(t, t);
figure(1);
contourf(x, y, v_vec{1}(x,y));
grid on;
axis equal;
figure(2);
contourf(x, y, v_vec{2}(x,y));
grid on;
axis equal;
figure(3);
contourf(x, y, v_vec{3}(x,y));
grid on;
axis equal;