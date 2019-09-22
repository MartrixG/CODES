#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<string>
#include<iostream>
using namespace std;
#define ll long long
#define re register
#define gc getchar
#define pc putchar
#define cs const

inline
int getint(){
	re int num;
	re char c;
	re bool f=0;
	while(!isdigit(c=gc()))f^=c=='-';num=c^48;
	while(isdigit(c=gc()))num=(num<<1)+(num<<3)+(c^48);
	return f?-num:num;
}

inline
void outint(int a){
	static char ch[13];
	if(a==0)pc('0');
	if(a<0)pc('-'),a=-a;
	while(a)ch[++ch[0]]=a-a/10*10,a/=10;
	while(ch[0])pc(ch[ch[0]--]^48);
}

typedef struct splay_node *point;
struct splay_node{
	point son[2],fa;
	int val,cnt,siz;
	splay_node(int _val=0){
		son[0]=son[1]=NULL;
		fa=NULL;
		val=_val;
		siz=cnt=1;
	}
	
	point &lc(){return son[0];}
	point &rc(){return son[1];}
	bool which(){return fa->rc()==this;}
	void update(){
		siz=(son[0]?son[0]->siz:0)+(son[1]?son[1]->siz:0)+cnt;
	}
	void init(){
		lc()=rc()=fa=NULL;
		siz=cnt=1;
	}
	void clear(){
		lc()=rc()=fa=NULL;
		val=siz=cnt=0;
	}
};

struct SPLAY{
	point root;
	SPLAY():root(NULL){}
	
	void Rotate(point now){
		point Fa=now->fa;
		bool pos=!now->which();
		Fa->son[!pos]=now->son[pos];
		if(now->son[pos])now->son[pos]->fa=Fa;
		if(now->fa=Fa->fa)now->fa->son[Fa->which()]=now;
		Fa->fa=now;
		now->son[pos]=Fa;
		Fa->update();
		now->update();
		if(now->fa==NULL)root=now;
	}
	
	void Splay(point now,point to=NULL){
		for(point Fa=now->fa;(Fa=now->fa)!=to;Rotate(now))
		if(Fa->fa!=to)Rotate(now->which()==Fa->which()?Fa:now);
	}
	
	void Insert(cs int &key){
		if(!root){
			root=(point)malloc(sizeof(splay_node));
			root->init();
			root->val=key;
			return ;
		}
		point now=root,Fa;
		for(;;Fa=now,now=now->son[key>now->val]){
			if(now==NULL){
				now=(point)malloc(sizeof(splay_node));
				now->init();
				now->fa=Fa;
				now->val=key;
				Fa->son[key>Fa->val]=now;
				return Splay(now);
			}
			if(now->val==key){
				++now->cnt;
				Splay(now);
				return ;
			}
		}
	}
	
	point find(cs int &key){
		point now=root;
		while(now!=NULL&&now->val!=key)now=now->son[key>now->val];
		if(now!=NULL)Splay(now);
		return now;
	}
	
	point pre_pos(cs int &key){
		point now=find(key);
		re bool flag=false;
		if(now==NULL){
			Insert(key);
			now=root;
			flag=true;
		}
		if(now->lc()==NULL){
			if(flag)Delete(key);
			return NULL;
		}
		for(now=now->son[0];now->son[1];now=now->son[1]);
		if(flag)Delete(key);
		return now;
	}
	
	void Delete(cs int &key){
		point now=find(key);
		if(now==NULL)return ;
		if(now->cnt>1){
			--now->cnt;
			--now->siz;
			return ;
		}
		if(now->lc()==NULL&&now->rc()==NULL){
			free(now);
			root=NULL;
			return ;
		}
		if(now->lc()==NULL){
			root=now->rc();
			now->rc()->fa=NULL;
			free(now);
			return ;
		}
		if(now->rc()==NULL){
			root=now->lc();
			now->lc()->fa=NULL;
			free(now);
			return ;
		}
		point res_pre=pre_pos(now->val);
		point res=root;
		Splay(res_pre);
		
		root->rc()=res->rc();
		res->rc()->fa=root;
		root->update();
	}
	
	int querykth(int key){
		re int ans=0;
		re int res;
		point now=root;
		while(true){
			if(now->lc()&&key<=now->lc()->siz){
				now=now->lc();
				continue;
			}
			res=(now->lc()?now->lc()->siz:0)+now->cnt;
			if(key<=res)return now->val;
			key-=res;
			now=now->rc();
		}
	}
	
	int queryrank(cs int &key){
		point now=find(key);
		re bool flag=false;
		if(now==NULL){
			Insert(key);
			flag=true;
		}
		int ans=root->siz;
		if(root->rc())ans-=root->rc()->siz;
		ans-=root->cnt;
		if(flag)Delete(key);
		return ans;
	}
	
	int querypre(cs int &key){
		point now=find(key);
		bool flag=false;
		if(now==NULL){
			Insert(key);
			now=root;
			flag=true;
		}
		if(now->lc()==NULL){
			if(flag)Delete(key);
			return -1;
		}
		for(now=now->lc();now->rc();now=now->rc());
		if(flag)Delete(key);
		return now->val;
	}
	
	int querysuf(cs int &key){
		point now=find(key);
		bool flag=false;
		if(now==NULL){
			Insert(key);
			now=root;
			flag=true;
		}
		if(now->rc()==NULL){
			if(flag)Delete(key);
			return -1;
		}
		for(now=now->rc();now->lc();now=now->lc());
		if(flag)Delete(key);
		return now->val;
	}
	
}splay;

int n;
signed main(){
	n=getint();
	while(n--){
		int op=getint(),x=getint();
		switch(op){
			case 0:{splay.Insert(x);break;}
			case 1:{splay.Delete(x);break;}
			case 2:{outint(splay.querykth(x)),pc('\n');break;}
			case 3:{outint(splay.queryrank(x)),pc('\n');break;}
			case 4:{outint(splay.querypre(x)),pc('\n');break;}
			case 5:{outint(splay.querysuf(x)),pc('\n');break;}
		}
	}
	return 0;
} 