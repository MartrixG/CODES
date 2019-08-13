#include<bits/stdc++.h>
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))
#define LL long long
#define swap(x,y) (x^=y,y^=x,x^=y)
#define tc() (A==B&&(B=(A=ff)+fread(ff,1,100000,stdin),A==B)?EOF:*A++)
#define pc(ch) (pp_<100000?pp[pp_++]=(ch):(fwrite(pp,1,100000,stdout),pp[(pp_=0)++]=(ch)))
#define N 100000
int pp_=0;char ff[100000],*A=ff,*B=ff,pp[100000];
using namespace std;
int n,m,rt;
struct splay
{
	int Son[2],Size,Father,flag;
}node[N+5];
inline void read(int &x)
{
	x=0;int f=1;char ch;
	while(!isdigit(ch=tc())) f=ch^'-'?1:-1;
	while(x=(x<<3)+(x<<1)+ch-'0',isdigit(ch=tc()));
	x*=f;
}
inline void write(int x)
{
	if(x<0) pc('-'),x=-x;
	if(x>9) write(x/10);
	pc(x%10+'0');
}
inline void PushUp(int x)
{
	node[x].Size=node[node[x].Son[0]].Size+node[node[x].Son[1]].Size+1;
}
inline void PushDown(int x)//下推翻转标记
{
	if(node[x].flag) swap(node[x].Son[0],node[x].Son[1]),node[node[x].Son[0]].flag^=1,node[node[x].Son[1]].flag^=1,node[x].flag=0;//如果当前节点有翻转标记，那么交换其左右儿子，更新其左右儿子的翻转标记，然后清空当前节点的翻转标记
}
inline void Build(int l,int r,int &x)//一个建树的过程，是不是很像线段树？
{
	node[x=l+r>>1].Size=1;//先记录当前节点的编号和子树大小
	if(l<x) Build(l,x-1,node[x].Son[0]),node[node[x].Son[0]].Father=x;//如果当前节点左边还有元素，那么就继续对其左儿子建树
	if(x<r) Build(x+1,r,node[x].Son[1]),node[node[x].Son[1]].Father=x;//如果当前节点右边还有元素，那么就继续对其右儿子建树
	PushUp(x);//更新节点信息
}
inline int Which(int x)//判断当前节点是父亲的哪一个儿子
{
	return node[node[x].Father].Son[1]==x;
}
inline void Rotate(int x,int &k)//旋转操作
{
	int fa=node[x].Father,grandpa=node[fa].Father,d=Which(x);
	if(fa^k) node[grandpa].Son[Which(fa)]=x;
	else k=x;
	node[x].Father=grandpa,node[fa].Son[d]=node[x].Son[d^1],node[node[x].Son[d^1]].Father=fa,node[x].Son[d^1]=fa,node[fa].Father=x,PushUp(fa),PushUp(x); 
}
inline void Splay(int x,int &k)//不断将一个元素旋转至目标位置
{
	for(int fa=node[x].Father;x^k;fa=node[x].Father)
	{
		if(fa^k) Rotate(Which(fa)^Which(x)?x:fa,k);
		Rotate(x,k);
	}
}
inline int get_val(int pos)//求出中序遍历到的顺序为pos的节点的值
{
	int x=rt;
	while(x)
	{
		PushDown(x);//先下推标记，然后再操作
		if(node[node[x].Son[0]].Size==pos) return x;//如果当前节点中序遍历到的顺序等于pos，就返回当前节点的值
		if(node[node[x].Son[0]].Size>pos) x=node[x].Son[0];//如果当前节点左子树被中序遍历到的顺序大于pos，就访问当前节点的左子树
		else pos-=node[node[x].Son[0]].Size+1,x=node[x].Son[1];//否则，更新pos，访问右子树
	}
}
inline void rever(int x,int y)//翻转一个区间，具体操作见上面的解析
{
	int l=get_val(x-1),r=get_val(y+1);
	Splay(l,rt),Splay(r,node[rt].Son[1]),node[node[node[rt].Son[1]].Son[0]].flag^=1;
}
int main()
{
	register int i;int x,y;
	for(read(n),Build(1,n+2,rt),read(m);m;--m) read(x),read(y),rever(x,y);
	for(i=1;i<=n;++i) write(get_val(i)-1),pc(' ');//由于我们用中序遍历到的顺序为2~n+1的节点来表示序列中第1~n个元素，所以输出时将答案减1
	return fwrite(pp,1,pp_,stdout),0;
}
