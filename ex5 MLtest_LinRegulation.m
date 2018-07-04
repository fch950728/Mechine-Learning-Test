%�ߴκ������Իع�����
%����ģ�������ع�
clc
clear

x = load('ex5Data/ex5Linx.dat');
y = load('ex5Data/ex5Liny.dat');


plot(x,y,'o','MarkerEdgeColor','b','MarkerFaceColor','r')
hold on%����һ�λ�ͼǰ����holdon���ܱ�֤��������ͬһ��ͼ��

m = length(x);
x = [ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
[m,n] = size(x);
n = n-1;
lambda = [0,1,10];
colortype = {'g','b','r'};
con = diag([0;ones(n,1)]);%diag���÷��ǰ�һ���������������������ݷ��ڶԽ�����
theta = zeros(n+1,3);
xrange = linspace(min(x(:,2)),max(x(:,2)))';

% ʹ��normal equation�㷨
for i = 1:3
    theta(:,i) = inv(x'*x+lambda(i).*con)*x'*y;
    norm_theta = norm(theta);%��ƽ��
    yrange = [ones(size(xrange)),xrange,xrange.^2,xrange.^3,xrange.^4,xrange.^5]*theta(:,i);
    plot(xrange',yrange,char(colortype(i)))
    hold on
end
legend('traning data','\lambda = 0','\lambda = 1','\lambda = 10')
hold off
    
    