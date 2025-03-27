addpath('/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/examples/dengwirda-mesh2d-99e0b24/');
initmsh()

%% L-Shape (no holes)
% Nodes
l_nodes = [0, 0; 2, 0; 2, 1; 1, 1; 1, 2; 0, 2];
% Edges (closed loop)
l_edges = [1,2; 2,3; 3,4; 4,5; 5,6; 6,1];
% No holes
hole_points = [];
% Triangulate
hdata.hmax = 0.1;
[vert,etri, ...
tria,tnum] = refine2(l_nodes, l_edges);
% Plot
figure;
triplot(tria, vert(:,1), vert(:,2));
title('L-Shape Triangulation');
axis equal;


%% Pentagon with Pentagon Hole
% Outer pentagon (counter-clockwise)
theta = linspace(0, 2*pi, 6); theta(end) = [];
outer_nodes = [cos(theta)', sin(theta)'];
outer_edges = [1:5; 2:5, 1]';  % Close the loop

% Inner pentagon hole (clockwise)
theta_rev = flip(theta);  % Reverse theta for clockwise
inner_nodes = 0.4 * [cos(theta_rev + pi)', sin(theta_rev + pi)'];
inner_edges = [6:10; 7:10, 6]' + 5;  % Offset by outer node count

% Combine nodes/edges
nodes = [outer_nodes; inner_nodes];
edges = [outer_edges; inner_edges];
% Hole point (centroid of inner pentagon)
hole_points = mean(inner_nodes);

% Triangulate
[vert,etri, ...
tria,tnum] = refine2(nodes, edges, [], hole_points);
% Plot
figure;
triplot(tria, vert(:,1), vert(:,2));
title('Pentagon with Hole Triangulation');
axis equal;
