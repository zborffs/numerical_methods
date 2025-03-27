% MATLAB script to mesh the given shapes using distmesh2d
addpath '/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/examples/distmesh/distmesh'

clc; clear; close all;

%% Parameters
h = 0.05; % Mesh size

%
% fd=@(p) sqrt(sum(p.^2,2))-1;
% [p,t]=distmesh2d(fd,@huniform,0.2,[-1,-1;1,1],[]);
% plt(t_L, p_L(:,1), p_L(:,2)); title('Example'); axis equal;

%% L-shaped Region
% fd_L = @(p) ddiff(drectangle(p,-1,1,-1,1), drectangle(p,0,1,0,1));
% pfix_L = [-1 -1; -1 1; 1 1; 1 0; 0 0; 0 1; 0 -1];
% [p_L,t_L] = distmesh2d(fd_L, @huniform, h, [-1,-1;1,1], pfix_L);

%% Pentagon with Inner Star
pent = [cos((0:5)*2*pi/5); sin((0:5)*2*pi/5)]';
star = 0.5*[cos((0:5)*4*pi/5+pi/5); sin((0:5)*4*pi/5+pi/5)]';
fd_P = @(p) dunion(dpoly(p, pent), dpoly(p, star));
[p_P,t_P] = distmesh2d(fd_P,@huniform,h,[-1,-1;1,1],[]);
% pent = [cos((0:5)*2*pi/5); sin((0:5)*2*pi/5)]';
% star = 0.5*[cos((0:5)*4*pi/5+pi/5); sin((0:5)*4*pi/5+pi/5)]';
% fd_P = @(p) ddiff(dsegment(p, pent), dsegment(p, star));
% [p_P,t_P] = distmesh2d(fd_P,@huniform,h,[-1,-1;1,1],[]);

%% Semi-circle with Two Holes
% fd_S = @(p) ddiff(dcircle(p,0,0,1), dunion(dunion(dcircle(p,0.5,-0.5,0.2), dcircle(p,-0.5,-0.5,0.2)), drectangle(p, -1, 1, 0, 1)));
% pfix_S = [0 0; -1 0; 1 0];
% [p_S,t_S] = distmesh2d(fd_S,@huniform,h,[-1,-1;1,1],pfix_S);