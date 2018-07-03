clear
clc

x = load('ex4Data/ex4x.dat');
y = load('ex4Data/ex4y.dat');
m = length(x);
x = [ones(m,1),x];

pos = find(y==1);%找到y=1的那些函数值
neg = find(y==0);

plot(x(pos,2),x(pos,3),'+');hold on
plot(x(neg,2),x(neg,3),'o')

theta = zeros(size(x(1,:)))';
MAX_LTR = 20;%一般15-20次即收敛
h = inline('1.0./(1.0+exp(-z))','z');%构造一个函数用以表示sigmod函数

for i = 1:MAX_LTR
    htheta = h(x*theta);
    err = htheta - y;
    deltaJ = (1/m).*x'*err;%损失函数的导数值
    H = (1/m).*x'*diag(htheta)*diag(1-htheta)*x;%Hessian矩阵的向量构造式11
    
    J(i) = (1/m)*sum(-y.*log(htheta)-(1-y).*(log(1-htheta)));%逻辑回归的损失函数表示式
    theta = theta - H\deltaJ;
end

theta
prob = 1-h([1,20,80]*theta)
plot_x = [min(x(:,2))-2,max(x(:,2))+2];
plot_y = (-1./theta(3)).*(theta(1)+theta(2)*plot_x);%由于分界线上概率为0.5，带入sigmod式可以算出x2和x1的关系，plot_y其实表示的是x2
plot(plot_x,plot_y)
xlabel('testmark1');ylabel('testmark2');
legend('Admitted','Not admitted','Decision Boundary')
hold off

%%%%%绘制迭代的次数和损失函数的关系
figure 
plot(0:MAX_LTR-1,J,'o--','MarkerFaceColor','r','MarkerSize',8)
xlabel('Iteration');ylabel('J');
J
