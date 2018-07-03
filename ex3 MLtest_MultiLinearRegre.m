clear
clc

x = load('ex3Data/ex3x.dat');
y = load('ex3Data/ex3y.dat');

m = length(x);
x = [ones(m,1),x];

figure;
plot(x,y,'o')

% 尺度变换
sigma = std(x);%标准差
mu = mean(x);%均值
x(:,2) = (x(:,2) - mu(2))./sigma(2);
x(:,3) = (x(:,3) - mu(3))./sigma(3); 

MAX_LTR = 50;
alfa = [0.01,0.03,0.1,0.3,1,1.3];%尝试不同的学习率
plotstyle = {'b','r','g','k','b--','r--'};

figure;
for i = (1:6)%尝试不同学习率的收敛速度
    theta = zeros(size(x(1,:)))';
    Jtheta = zeros(MAX_LTR,1);
    for j = (1:MAX_LTR)
        grad = (1/m).*x'*(x*theta-y);
        err = x*theta - y;
        Jtheta(j) = (1/(2*m)).*err'*err;%误差函数
        theta = theta - alfa(i).*grad;%梯度下降
    end 
    plot(0:49,Jtheta(1:50),char(plotstyle(i)),'LineWidth',2)
    hold on
end
legend('0.01','0,03','0.1','0.3','1','1.3');
xlabel('Number of iterations')
ylabel('Cost function')

% 另一种方法
x = load('ex3Data/ex3x.dat');
y = load('ex3Data/ex3y.dat');
x = [ones(size(x,1),1),x];
theta_norequ = inv((x'*x))*x'*y%参数预测
price_norequ = [1 1650 3]*theta_norequ