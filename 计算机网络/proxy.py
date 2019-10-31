import socket
import sys
import threading
import time
import traceback
import os
import re
import hashlib

MAX_HEADER = 4096
RECV_SIZE = 512

BAN_LIST = [
    b'jwes.hit.edu.cn'
]

CHANGE_LIST = {
    b'www.hit.edu.cn' : b'studyathit.hit.edu.cn'
}

USER_LIST = [
    '127.0.0.1'
]

c = {}

def get_header(string, name):
    decode = string.decode('UTF-8')
    header = re.compile(name+r'.*', re.IGNORECASE)
    match = header.search(decode)
    if match:
        head = match.group()
        replace = re.compile(r'\r')
        head = replace.sub('', head)
        return head.encode('UTF-8')
    else:
        return None

def trans_host(raw_host):
    header = raw_host.decode('UTF-8', 'ignore')
    groups = header.split(":")
    host = groups[1].encode('UTF-8')
    if len(groups) > 2:
        port = int(groups[2])
    else:
        port = 80
    return host, port

def split_header(string):
    return string.split(b'\r\n\r\n')[0]

def recv_body(conn, base, size):
    if size == -1:
        while(base[-7:] != b'\r\n0\r\n\r\n'):
            base += conn.recv(RECV_SIZE)
    else:
        while len(base) < size:
            base += conn.recv(RECV_SIZE)
    return base

def check_cache(cache, url):
    hl = hashlib.md5()
    hl.update(url)
    url = hl.hexdigest()
    if cache.__contains__(url):
        return True
    else:
        return False

def write_cache(cache, url, timestamp, body):
    hl = hashlib.md5()
    hl.update(url)
    url = hl.hexdigest()
    cache[url] = timestamp
    file = open('计算机网络\dict.txt', 'a')
    file.write(url+'::'+timestamp+'\n')
    file.close()
    file = open('计算机网络\cache\\'+url, 'wb')
    file.write(body)
    file.close()

def load_body(cache, url):
    hl = hashlib.md5()
    hl.update(url)
    url = hl.hexdigest()
    for entry in os.listdir('计算机网络\cache'):
        if(entry == url):
            file = open('计算机网络\cache\\'+entry, 'rb')
            return file.read()


def thread_proxy(client, addr, cache, banlist, changelist, userlist):
    thread_name = threading.currentThread().name
    #监测是否ban IP地址
    if userlist != None:
        if userlist.count(addr[0]) != 0:
            print("%sThis client is banned!"%(thread_name))
            client.close()
            return
    #尝试接受客户端发送的requset
    try:
        request = client.recv(MAX_HEADER)
    except:
        print("%sTime out!"%(thread_name))
        client.close()
        return
    #获得初始的host
    raw_host = get_header(request, "Host").replace(b' ', b'')
    url = get_header(request, 'get').split(b' ')[1]
    
    if not raw_host:
        print("%sHost request error%s"%(thread_name, str(addr)))
        client.close()
        return
    
    host, port = trans_host(raw_host)
    print("%sGET:%s:%s"%(thread_name, url, str(port)))

    #钓鱼
    if changelist != None:
        if changelist.__contains__(host):
            host = changelist[host]
            print("%sHost has change to %s"%(thread_name, host))
    #禁止访问的host
    if banlist != None:
        if banlist.count(host) != 0:
            print("%sThis host is banned"%(thread_name))
            client.close()
            return
    #建立到服务器的连接
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.settimeout(10)
    try:
        server.connect((host, port))
    except socket.timeout:
        print("%sTime out!"%(thread_name))
        server.close()
        client.close()
        return
    
    #检查缓存
    if check_cache(cache, url):
        #修改request 监测是否变化
        url_md5 = hashlib.md5().update(url).hexdigest()
        modify = '\r\nIf-Modified-Since:'+cache[url_md5]+'\r\n\r\n'
        newrequest = request
        newrequest = newrequest.replace(b'\r\n\r\n', modify.encode('UTF-8'))
        server.sendall(newrequest)
        response = server.recv(MAX_HEADER)
        responseHeader = split_header(response)
        flag = get_header(responseHeader, 'HTTP/1.1').split(b' ')[1]
        if flag == b'304':
            print("%sCache hit!!"%(thread_name))
            response = load_body(cache, url)
            client.sendall(response)

            server.close()
            client.close()
            return
    
    #未命中或者已经发生修改发送原的request
    server.sendall(request)

    response = server.recv(RECV_SIZE)
    responseHeader = split_header(response)

    if len(responseHeader) < len(response) - 4:
        content_size = get_header(responseHeader, 'content-length')
        if content_size:
            size = int(content_size.split(b':')[1]) + 4 + len(responseHeader)
        else:
            size = -1
        response = recv_body(server, response, size)
    client.sendall(response)
    time = get_header(responseHeader, 'Last-Modified')
    if time != None:
        #如果含有Last-Modified说明可被缓存
        time = time.split(b': ')[1].decode('UTF-8')
        write_cache(cache, url, time, response)

    server.close()
    client.close()

def thread_server(myserver):
    while True:
        conn, addr = myserver.accept()
        conn.settimeout(10)
        thread_p = threading.Thread(target=thread_proxy, args=(conn, addr, c, None , None, None))
        thread_p.setDaemon(True)
        thread_p.start()

def main(port=8000):
    try:
        myserver = socket.socket()
        myserver.bind(('127.0.0.1', port))
        myserver.listen(1024)
        thread_s = threading.Thread(target=thread_server, args=(myserver,))
        thread_s.setDaemon(True)
        thread_s.start()
        while True:
            time.sleep(1000)
    except KeyboardInterrupt:
        print("sys exit")
    finally:
        myserver.close()

def loadCache():
    file = open('计算机网络\dict.txt', 'r')
    line = file.readline()
    while line:
        line = line.split('::')
        c[line[0]] = line[1][:-1]
        line = file.readline()
# 命令入口
if __name__ == '__main__':
    try:
        loadCache()
        print("Start proxy...")
        main()
    except Exception as e:
        print("error exit")
        traceback.print_exc()
    finally:
        print("end server")
    sys.exit(0)