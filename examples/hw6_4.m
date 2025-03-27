% MATLAB script to mesh the given shapes using MESH2D
addpath '/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/examples'/dengwirda-mesh2d-99e0b24/

%------------------------------------------- setup geometry

node = [                % list of xy "node" coordinates
    0, 0                % outer square
    1, 0
    1, 0.5
    0.5, 0.5
    0.5, 1                % inner square
    0, 1 
] ;

edge = [                % list of "edges" between nodes
    1, 2                % outer square
    2, 3
    3, 4
    4, 5
    5, 6                % inner square
    6, 1 
] ;

%------------------------------------------- call mesh-gen.
   [vert,etri, ...
    tria,tnum] = refine2(node,edge) ;

%------------------------------------------- draw tria-mesh
    figure;
    patch('faces',tria(:,1:3),'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2,.2,.2]) ;
    hold on; axis image off;
    patch('faces',edge(:,1:2),'vertices',node, ...
        'facecolor','w', ...
        'edgecolor',[.1,.1,.1], ...
        'linewidth',1.5) ;

%%
clear
clc

w = exp(i * 2 * pi / 5);
node = [
    real(i), imag(i)
    real(w*i), imag(w*i)
    real(w^2*i), imag(w^2*i)
    real(w^3*i), imag(w^3*i)
    real(w^4*i), imag(w^4*i)
    real(-i/3), imag(-i/3)
    real(w*-i/3), imag(w*-i/3)
    real(w^2*-i/3), imag(w^2*-i/3)
    real(w^3*-i/3), imag(w^3*-i/3)
    real(w^4*-i/3), imag(w^4*-i/3)
];


edge = [
    1, 2
    2, 3
    3, 4
    4, 5
    5, 1
    6, 7
    7, 8
    8, 9
    9, 10
    10, 6
]

[vert,etri, ...
    tria,tnum] = refine2(node,edge) ;

%------------------------------------------- draw tria-mesh
    figure;
    patch('faces',tria(:,1:3),'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2,.2,.2]) ;
    hold on; axis image off;
    patch('faces',edge(:,1:2),'vertices',node, ...
        'facecolor','w', ...
        'edgecolor',[.1,.1,.1], ...
        'linewidth',1.5) ;


%%
clear
clc

w = exp(i * 2 * pi / 5);
node = [
    real(i), imag(i)
    real(w*i), imag(w*i)
    real(w^2*i), imag(w^2*i)
    real(w^3*i), imag(w^3*i)
    real(w^4*i), imag(w^4*i)
    real(-i/3), imag(-i/3)
    real(w*-i/3), imag(w*-i/3)
    real(w^2*-i/3), imag(w^2*-i/3)
    real(w^3*-i/3), imag(w^3*-i/3)
    real(w^4*-i/3), imag(w^4*-i/3)
];


edge = [
    1, 2
    2, 3
    3, 4
    4, 5
    5, 1
    6, 7
    7, 8
    8, 9
    9, 10
    10, 6
]

[vert,etri, ...
    tria,tnum] = refine2(node,edge) ;

%------------------------------------------- draw tria-mesh
    figure;
    patch('faces',tria(:,1:3),'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2,.2,.2]) ;
    hold on; axis image off;
    patch('faces',edge(:,1:2),'vertices',node, ...
        'facecolor','w', ...
        'edgecolor',[.1,.1,.1], ...
        'linewidth',1.5) ;



%%
clear
clc



N = 100;
node = [];
edge = [];
for n = 1:N
    node = [node; cos(n / N * pi), sin(-n / N * pi)];
    edge = [edge; n, mod(n, N)+1];
end
for n = 1:N
    node = [node; -0.5 + 0.2 * cos(n / N * 2 * pi), -0.4 + 0.2 * sin(n / N * 2 * pi)];
    edge = [edge; N+n, mod(n, N)+N+1];
end
for n = 1:N
    node = [node; 0.5 + 0.2 * cos(n / N * 2 * pi), -0.4 + 0.2 * sin(n / N * 2 * pi)];
    edge = [edge; 2*N+n, mod(n, N)+2*N+1];
end




[vert,etri, ...
    tria,tnum] = refine2(node,edge) ;

%------------------------------------------- draw tria-mesh
    figure;
    patch('faces',tria(:,1:3),'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2,.2,.2]) ;
    hold on; axis image off;
    patch('faces',edge(:,1:2),'vertices',node, ...
        'facecolor','w', ...
        'edgecolor',[.1,.1,.1], ...
        'linewidth',1.5) ;