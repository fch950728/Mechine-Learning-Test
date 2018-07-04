%高次函数线性回归正则化
%该题模拟的是岭回归
clc
clear

x = load('ex5Data/ex5Linx.dat');
y = load('ex5Data/ex5Liny.dat');


plot(x,y,'o','MarkerEdgeColor','b','MarkerFaceColor','r')
hold on%在下一次绘图前调用holdon才能保证多个点绘在同一张图上

m = length(x);
x = [ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
[m,n] = size(x);
n = n-1;
lambda = [0,1,10];
colortype = {'g','b','r'};
con = diag([0;ones(n,1)]);%diag的用法是把一个行向量或列向量的数据放在对角线上
theta = zeros(n+1,3);
xrange = linspace(min(x(:,2)),max(x(:,2)))';

% 使用normal equation算法
for i = 1:3
    theta(:,i) = inv(x'*x+lambda(i).*con)*x'*y;
    norm_theta = norm(theta);%点平方
    yrange = [ones(size(xrange)),xrange,xrange.^2,xrange.^3,xrange.^4,xrange.^5]*theta(:,i);
    plot(xrange',yrange,char(colortype(i)))
    hold on
end
legend('traning data','\lambda = 0','\lambda = 1','\lambda = 10')
hold off
    
    