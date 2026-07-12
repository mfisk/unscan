#!/usr/bin/env python3
"""Convert HTML to PDF using Chrome DevTools Protocol — landscape, custom size."""
import json, sys, subprocess, time, socket, base64, http.client, os, websocket

def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main():
    if len(sys.argv) < 2:
        print("Usage: html2pdf_cdp.py <input.html> [output.pdf]", file=sys.stderr)
        sys.exit(1)

    inp = os.path.abspath(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else inp.rsplit('.', 1)[0] + '.pdf'
    out = os.path.abspath(out)

    port = find_free_port()
    chrome = subprocess.Popen([
        'google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox',
        '--host-resolver-rules=MAP * ~NOTFOUND',
        f'--remote-debugging-port={port}',
        '--remote-debugging-address=127.0.0.1',
        '--remote-allow-origins=*',
        'about:blank'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(30):
        try:
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
            conn.request('GET', '/json/version')
            if conn.getresponse().status == 200: break
        except: pass
        time.sleep(0.5)
    else:
        chrome.kill(); sys.exit(1)

    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', '/json/list')
    tabs = json.loads(conn.getresponse().read())
    ws_url = tabs[0]['webSocketDebuggerUrl']
    ws = websocket.create_connection(ws_url, timeout=600)

    msg_id = 1
    def send_cmd(method, params=None):
        nonlocal msg_id
        msg = {'id': msg_id, 'method': method}
        if params: msg['params'] = params
        ws.send(json.dumps(msg))
        while True:
            resp = json.loads(ws.recv())
            if resp.get('id') == msg_id:
                msg_id += 1
                return resp

    send_cmd('Page.enable')
    send_cmd('Page.navigate', {'url': f'file://{inp}'})

    # Wait for load event
    deadline = time.time() + 300
    while time.time() < deadline:
        ws.settimeout(max(1, deadline - time.time()))
        try:
            msg = json.loads(ws.recv())
            if msg.get('method') == 'Page.loadEventFired': break
        except websocket.WebSocketTimeoutException: break

    time.sleep(5)

    # Landscape letter: 11in x 8.5in
    result = send_cmd('Page.printToPDF', {
        'landscape': True,
        'paperWidth': 14,
        'paperHeight': 8.5,
        'marginTop': 0.3,
        'marginBottom': 0.3,
        'marginLeft': 0.3,
        'marginRight': 0.3,
        'printBackground': True,
        'preferCSSPageSize': False,
        'displayHeaderFooter': False,
    })

    if 'result' not in result or 'data' not in result.get('result', {}):
        print(f"CDP error: {json.dumps(result)}", file=sys.stderr)
        ws.close(); chrome.kill(); sys.exit(1)

    pdf_data = base64.b64decode(result['result']['data'])
    with open(out, 'wb') as f:
        f.write(pdf_data)

    ws.close()
    chrome.kill()
    chrome.wait()
    print(f"{out} ({len(pdf_data)//1024}K)")

if __name__ == '__main__':
    main()
