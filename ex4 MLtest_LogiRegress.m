clear
clc

x = load('ex4Data/ex4x.dat');
y = load('ex4Data/ex4y.dat');
m = length(x);
x = [ones(m,1),x];

pos = find(y==1);%�ҵ�y=1����Щ����ֵ
neg = find(y==0);

plot(x(pos,2),x(pos,3),'+');hold on
plot(x(neg,2),x(neg,3),'o')

theta = zeros(size(x(1,:)))';
MAX_LTR = 20;%һ��15-20�μ�����
h = inline('1.0./(1.0+exp(-z))','z');%����һ���������Ա�ʾsigmod����

for i = 1:MAX_LTR
    htheta = h(x*theta);
    err = htheta - y;
    deltaJ = (1/m).*x'*err;%��ʧ�����ĵ���ֵ
    H = (1/m).*x'*diag(htheta)*diag(1-htheta)*x;%Hessian�������������ʽ11
    
    J(i) = (1/m)*sum(-y.*log(htheta)-(1-y).*(log(1-htheta)));%�߼��ع����ʧ������ʾʽ
    theta = theta - H\deltaJ;
end

theta
prob = 1-h([1,20,80]*theta)
plot_x = [min(x(:,2))-2,max(x(:,2))+2];
plot_y = (-1./theta(3)).*(theta(1)+theta(2)*plot_x);%���ڷֽ����ϸ���Ϊ0.5������sigmodʽ�������x2��x1�Ĺ�ϵ��plot_y��ʵ��ʾ����x2
plot(plot_x,plot_y)
xlabel('testmark1');ylabel('testmark2');
legend('Admitted','Not admitted','Decision Boundary')
hold off

%%%%%���Ƶ����Ĵ�������ʧ�����Ĺ�ϵ
figure 
plot(0:MAX_LTR-1,J,'o--','MarkerFaceColor','r','MarkerSize',8)
xlabel('Iteration');ylabel('J');
J
