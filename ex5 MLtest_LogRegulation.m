%���߼�����������������������
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
x = map_feature(u,v);%��Ϊ28ά����ֻҪ��֤u��v��ͬ������������
[m,n] = size(x);


MAX_LTR = 20;
h = inline('1.0./(1.0+exp(-z))','z');%����sigmoid����

lambda = 10;%�������趨����ֵ
con = diag([0;ones(n-1,1)]);

theta = zeros(size(x(1,:)))';
    for j = 1:MAX_LTR%���㲽�����ͨ�߼��ع����ƣ�ֻ����Ҫ���������
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
         z(i,j) = map_feature(u(i),v(j))*theta;%ע��z����28ά�任�õ���
     end
 end
 
 z = z';
 
 contour(u,v,z,[0,0],'LineWidth',2)%���������ߣ��ڶ�άƽ��u-v�л�������z��������[0,0]��ָ����ֵz��0��0֮��ĵȸ���
 legend('y=1','y=0','Decision boundary')
 hold off
 
 

    
    