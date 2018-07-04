%在逻辑函数中添加正则项避免过拟合
clc
clear

x = load('ex5Data/ex5Logx.dat');
y = load('ex5Data/ex5Logy.dat');

m = length(x);
figure;

pos = find(y==1);
neg = find(y==0);

plot(x(pos,1),x(pos,2),'+');
hold on
plot(x(neg,1),x(neg,2),'o');

u = x(:,1);
v = x(:,2);
x = map_feature(u,v);%变为28维矩阵，只要保证u，v是同型向量就能用
[m,n] = size(x);


MAX_LTR = 20;
h = inline('1.0./(1.0+exp(-z))','z');%构造sigmoid函数

lambda = 10;%在这里设定参数值
con = diag([0;ones(n-1,1)]);

theta = zeros(size(x(1,:)))';
    for j = 1:MAX_LTR%计算步骤和普通逻辑回归类似，只是需要添加正则项
        z = h(x*theta); 
        err = z - y;
        deltaJ = (1/m).*x'*err + (lambda/m).*con*theta;
        H = (1/m).*x'*diag(z)*diag(1-z)*x + lambda/m.*con;
        theta = theta - H\deltaJ;
    end
    
 hold on 
 u = linspace(-1,1.5,200);
 v = linspace(-1,1.5,200);
 
 z = zeros(length(u),length(v));
 for i = 1:length(u)
     for j = 1:length(v)
         z(i,j) = map_feature(u(i),v(j))*theta;%注意z是由28维变换得到的
     end
 end
 
 z = z';
 
 contour(u,v,z,[0,0],'LineWidth',2)%绘制轮廓线，在二维平面u-v中绘制曲面z的轮廓，[0,0]是指函数值z在0和0之间的等高线
 legend('y=1','y=0','Decision boundary')
 hold off
 
 

    
    