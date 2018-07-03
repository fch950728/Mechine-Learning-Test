x = load('ex2Data/ex2x.dat');
y = load('ex2Data/ex2y.dat');

plot(x,y,'o');
ylabel('Height in meters')
xlabel('Age in years')

m = length(y);
x = [ones(m,1) x];%相当于指定x0等于1 ones产生一个只有1的矩阵


theta = zeros(size(x(1,:)))';%指定系数矩阵
literator = 1 ;
MAX_LIT = 1500;%最大迭代次数
alfa = 0.07;%学习率

while  literator < MAX_LIT
    grad = (1/m).*x'*((x * theta) - y);%梯度下降法，注意这里使用矩阵乘法的方法来代替累加
    theta = theta - alfa.* grad;
    literator = literator + 1 ;
end 

hold on 
plot(x(:,2),x*theta,'-')
legend('Training data','Linear regression')

J_vals = zeros(100,100);%绘制3维坐标图来说明系数和损失函数的关系
theta0_vals = linspace(-3,3,100);
theta1_vals = linspace(-1,1,100);
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i);theta1_vals(j)];
        err = (x*t)-y;
        J_vals(i,j) = (1/(2*m))*err'*err;
    end 
end

J_vals = J_vals';
figure;
surf(theta0_vals,theta1_vals,J_vals)
xlabel('\theta_0');ylabel('\theta_1');

figure;
contour(theta0_vals,theta1_vals,J_vals,logspace(-2,2,15))%绘制等高线
xlabel('\theta_0');ylabel('\theta_1');
