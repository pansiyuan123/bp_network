#include<cstdio>
#include<algorithm>
#include<cstring>
#include<cmath>
#include<vector>
#include<windows.h>
#include<time.h>
#define random(x) (rand()%x)

using namespace std;                                //����˵д����Ҫ����дע�͵ĺ�ϰ��

vector <vector<double> > input;                     //ûÿһ�����ݵ�����

vector <vector<double> > label;                     //ÿһ�����ݵı�ǩ

vector <int> layer_nodesnum;                        //ÿһ���м����ڵ�

vector <vector<double> > output;                    //����Ľ������label���Ӧ

int trainnum,inputnum,layernum,labelnum;            //����ѵ�����ݣ�ÿһ�������м������룬�ܹ��м��㣬���Ӧ���м�����Ԫ

vector <double> learningrate;

double wishloss;

double sigmod_d(double x)
{
    double temp;
    temp = exp(0.0-x);
    return temp/((1.0+temp)*(1.0+temp));
}

double sigmod(double x)
{
    double temp;
    temp = exp(0.0-x);
    return 1.0/(1.0+temp);
}

void readfile()
{
    freopen("input.txt","r",stdin);
    scanf("%d %d %d",&trainnum,&inputnum,&labelnum);
    input.resize(trainnum);
    label.resize(trainnum);
    for(int i=0;i<trainnum;i++)
    {
        double temp;
        for(int j=0;j<inputnum;j++)
        {
            scanf("%lf",&temp);
            input[i].push_back(temp);
        }
        for(int j=0;j<labelnum;j++)
        {
            scanf("%lf",&temp);
            label[i].push_back(temp);
        }
    }
    scanf("%d",&layernum);
    for(int i=0;i<layernum;i++)
    {
        int temp;
        scanf("%d",&temp);
        layer_nodesnum.push_back(temp);
    }
    for(int i=0;i<layernum;i++)
    {
        double temp;
        scanf("%lf",&temp);
        learningrate.push_back(temp);
    }
    scanf("%lf",&wishloss);
    fclose(stdin);
}

enum type{inputlayer,hiddenlayer,outputlayer};

struct node
{
    vector <double> w;
    vector <double> w_change;
    double d;
    double d_change;
    type nodetype;
    double value;
};

vector <vector<node> > layer;                                    //ÿһ��Ľڵ����

void init()
{
    output.resize(trainnum);
    for(int i=0;i<trainnum;i++)
        for(int j=0;j<labelnum;j++)
        output[i].push_back(0.0);
    layer.resize(layernum);
    for(int i=0;i<layernum;i++)
    {
        for(int j=0;j<layer_nodesnum[i];j++)
        {
            node temp;
            if(i==0)
            {
                temp.nodetype = inputlayer;
            }
            else if(i==layernum-1)
            {
                temp.nodetype = outputlayer;
                temp.d = (double)random(100)/100.0;
                for(int k=0;k<layer_nodesnum[i-1];k++)
                {
                    temp.w.push_back((double)random(100)/100.0);
                    temp.w_change.push_back(0.0);
                }
            }
            else
            {
                temp.nodetype = hiddenlayer;
                temp.d = (double)random(100)/100.0;
                for(int k=0;k<layer_nodesnum[i-1];k++)
                {
                    temp.w.push_back((double)random(100)/100.0);
                    temp.w_change.push_back(0.0);
                }
            }
            layer[i].push_back(temp);
        }
    }
}

double compute_loss_sum(void)
{
    double sum = 0.0;
    for(int i=0;i<trainnum;i++)
        for(int j=0;j<labelnum;j++)
            sum=sum+(label[i][j]-output[i][j])*(label[i][j]-output[i][j]);
    sum = sum/2.0/trainnum;
    return sum;
}

double compute_loss(int k)
{
    double sum=0.0;
    for(int i=0;i<labelnum;i++)
    {
        sum = sum + (label[k][i]-output[k][i])*(label[k][i]-output[k][i]);
    }
    return sum/2.0;
}

void fw(int k)
{
    for(int i=0;i<layer[0].size();i++)
    {
        node temp = layer[0][i];
        temp.value = input[k][i];
        layer[0][i] = temp;
    }
    for(int i=1;i<layernum;i++)
    {
        for(int j=0;j<layer_nodesnum[i];j++)
        {
            node temp = layer[i][j];
            double tempp = 0.0;
            for(int p=0;p<layer_nodesnum[i-1];p++)
            {
                tempp = tempp + temp.w[p]*layer[i-1][p].value;
            }
            tempp = tempp + temp.d;
            temp.value = sigmod(tempp);
            layer[i][j] = temp;
        }
    }

    /*printf("%d :",k);
    for(int i=0;i<inputnum;i++)
        printf("%lf ",input[k][i]);
    printf("\n");
    for(int i=0;i<labelnum;i++)
    {
        output[k][i]=layer[layernum-1][i].value;
        printf("%lf ",output[k][i]);
    }
    printf("\n");
    Sleep(1000);*/
    for(int i=0;i<labelnum;i++)
        output[k][i]=layer[layernum-1][i].value;
}

//���򴫲�д�ļ�ֱ��Ѫ������������

void bw(int k)
{
    int current = layernum-1;
    for(int i=0;i<layer_nodesnum[current];i++)    //�ȼ������һ��
    {
        double change1 = -1.0*(label[k][i]-output[k][i]);         //�ܲв�����һ���i����Ԫ��Ӱ��
        double change2 = output[k][i]*(1.0-output[k][i]);         //��sigmoid֮ǰ��Ӱ��
        for(int j=0;j<layer_nodesnum[current-1];j++)
        {
            double change3 = layer[current-1][j].value;           //�Ե�j��������Ӱ��
            double change = change1*change2*change3;
            layer[current][i].w_change[j] = change;
        }
        layer[current][i].d_change = change1*change2;             //���Ϊ����һ�����Ӱ�죬Ҳ�������޸���ֵ��
    }
    current--;
    while(current>0)                       //�������һ��������,��0��Ϊ�����
    {
        for(int i=0;i<layer_nodesnum[current];i++)
        {
            double change1 = 0.0;
            for(int p=0;p<layer_nodesnum[current+1];p++)
            {
                change1 = change1+layer[current+1][p].d_change*layer[current+1][p].w[i];   //�Ѷ���������Ӱ��Ĳ���ҵ�Ӱ���ۼ�����
            }
            double change2 = layer[current][i].value*(1.0-layer[current][i].value);          //��sigmoid֮ǰ��Ӱ��
            for(int j=0;j<layer_nodesnum[current-1];j++)
            {
                double change3 = layer[current-1][j].value;                             //��һ�������ֵ
                double change = change1*change2*change3;
                layer[current][i].w_change[j] = change;
            }
            layer[current][i].d_change = change1*change2;
        }
        current--;
    }
    for(int i=1;i<layernum;i++)
    {
        for(int j=0;j<layer_nodesnum[i];j++)
        {
            for(int p=0;p<layer_nodesnum[i-1];p++)
            {
                layer[i][j].w[p] = layer[i][j].w[p] - learningrate[i]*layer[i][j].w_change[p];
            }
            layer[i][j].d = layer[i][j].d - learningrate[i]*layer[i][j].d_change;
        }
    }
}

/*void rengong_fuck(void)
{
    layer[1][0].w[0]=0.0543;
    layer[1][0].w[1]=0.0579;
    layer[1][0].d=-0.0703;
    layer[1][1].w[0]=-0.0291;
    layer[1][1].w[1]=0.0999;
    layer[1][1].d=-0.0939;
    layer[2][0].w[0]=0.0801;
    layer[2][0].w[1]=-0.0605;
    layer[2][0].d=-0.0109;
}*/

int main(void)
{
    srand((int)time(0));
    readfile();
    init();
    //rengong_fuck();                         //�ֶ������ʼȨ
    int ans=0;
    do
    {
        ans++;
        for(int i=0;i<trainnum;i++)
        {
            fw(i);
            bw(i);
           /* if(i==0&&ans==1)                         //Ϊ�˿�һ�µ�һ�ε��ݶ����
            {
               for(int i=1;i<layernum;i++)
                {
                    printf("the %d layer:\n",i);
                    for(int j=0;j<layer_nodesnum[i];j++)
                    {
                        for(int k=0;k<layer_nodesnum[i-1];k++)
                        {
                            printf("%lf ",layer[i][j].w_change[k]);
                        }
                        printf("%lf\n",layer[i][j].d_change);
                    }
                }
            }*/
        }
    }
    while(compute_loss_sum()>wishloss);
   /* for(int i=1;i<layernum;i++)                      //��������Ԫ�Ĳ���
    {
        printf("the %d layer:\n",i);
        for(int j=0;j<layer_nodesnum[i];j++)
        {
            for(int k=0;k<layer_nodesnum[i-1];k++)
            {
                printf("%lf ",layer[i][j].w[k]);
            }
            printf("%lf\n",layer[i][j].d);
        }
    }*/
    /*for(int i=0;i<trainnum;i++)                     //��ѵ���������������ݷ������
    {
        fw(i);
        for(int j=0;j<inputnum;j++)
            printf("%lf ",input[i][j]);
        printf("\n");
        for(int j=0;j<labelnum;j++)
            printf("%lf ",output[i][j]);
        printf("\n");
    }*/
    /*for(int i=0;i<trainnum;i++)                     //�����������������
    {
        for(int j=0;j<input[i].size();j++)
        {
            printf("%lf ",input[i][j]);
        }
        printf("\n");
        printf("%lf\n",label[i]);
    }
    for(int i=0;i<layer_nodesnum.size();i++)
        printf("%d " ,layer_nodesnum[i]);*/
    printf("%dtimes\n",ans);
    return 0;
}






