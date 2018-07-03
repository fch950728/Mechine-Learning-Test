x = load('ex2Data/ex2x.dat');
y = load('ex2Data/ex2y.dat');

plot(x,y,'o');
ylabel('Height in meters')
xlabel('Age in years')

m = length(y);
x = [ones(m,1) x];%�൱��ָ��x0����1 ones����һ��ֻ��1�ľ���


theta = zeros(size(x(1,:)))';%ָ��ϵ������
literator = 1 ;
MAX_LIT = 1500;%����������
alfa = 0.07;%ѧϰ��

while  literator < MAX_LIT
    grad = (1/m).*x'*((x * theta) - y);%�ݶ��½�����ע������ʹ�þ���˷��ķ����������ۼ�
    theta = theta - alfa.* grad;
    literator = literator + 1 ;
end 

hold on 
plot(x(:,2),x*theta,'-')
legend('Training data','Linear regression')

J_vals = zeros(100,100);%����3ά����ͼ��˵��ϵ������ʧ�����Ĺ�ϵ
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
contour(theta0_vals,theta1_vals,J_vals,logspace(-2,2,15))%���Ƶȸ���
xlabel('\theta_0');ylabel('\theta_1');
