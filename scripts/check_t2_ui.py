"""Smoke test da interface. Requer: pip install playwright; playwright install chromium."""
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]
server=subprocess.Popen([sys.executable,'app_t2.py'],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    for attempt in range(100):
        try:
            urllib.request.urlopen('http://127.0.0.1:5002',timeout=1).close()
            break
        except OSError:time.sleep(.1)
    else:raise RuntimeError('Servidor não iniciou.')
    with sync_playwright() as p:
        options={'headless':True,'args':['--no-sandbox']}
        if os.getenv('T2_CHROMIUM'):options['executable_path']=os.environ['T2_CHROMIUM']
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:5002')
        page.locator('[name=population]').fill('6')
        page.locator('[name=generations]').fill('4')
        page.locator('#advanced summary').click()
        page.locator('[name=medium_games]').fill('2')
        page.locator('#advanced summary').click()
        assert not page.locator('#advanced').evaluate('(e)=>e.open')
        assert not page.locator('#population-details').evaluate('(e)=>e.open')
        page.locator('#init').click()
        page.wait_for_function("!document.querySelector('#step').disabled")
        page.locator('#step').click()
        page.wait_for_selector('#population tr',state='attached')
        assert page.locator('#population tr').count()==6
        page.locator('#demo-details summary').click()
        page.locator('#demo').click()
        page.wait_for_function("!document.querySelector('#next').disabled")
        page.locator('#next').click()
        assert page.locator('#events li').count()==1
        page.locator('#new').click()
        page.locator('.cell').nth(0).click()
        page.wait_for_function("document.querySelectorAll('.cell:not(:empty)').length===2")
        page.locator('#mode').select_option('rede')
        page.locator('#starter').select_option('computer')
        page.wait_for_function("document.querySelectorAll('.cell:not(:empty)').length===1")
        page.locator('#evaluate').click()
        page.wait_for_selector('#evaluation p')
        assert page.locator('#evaluation p').count()==2
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors, errors
        browser.close()
        print('Interface OK: população, treino, replay, Minimax, rede, avaliação e largura móvel.')
finally:
    server.terminate()
    server.wait(timeout=10)
