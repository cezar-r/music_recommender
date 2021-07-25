import requests

scraper_url = 'http://api.scraperapi.com?api_key=800a43e7386a5a4f5801c8762c3a4aab&url='
base_url = 'https://www.listchallenges.com/500-random-musical-artists'

artists = ['360',
		'3pm',
		'5 seconds of summer',
		'the 88',
		'abba',
		'aero chord',
		'aerosmith',
		'afrojack',
		'ajr',
		'akon',
		'alicia keys',
		'alison wonderland',
		'all american rejects',
		'all star weekend',
		'amanda palmer',
		'amy shark',
		'anami vice',
		'anderson paak',
		'anthrax',
		'arcade fire',
		'ariana grande',
		'ariel pink',
		'aronchupa',
		'at the gates',
		'avicii',
		'avril lavinge',
		'backstreet boys',
		'bad suns',
		'ball park music',
		'the band perry',
		'the bangles',
		'barenaked ladies',
		'bastille',
		'bathory',
		'battle beast',
		'the beach boys']


def scrape_artists():
	for i in range(1, 14):
		if i > 1:
			response = requests.get(base_url + f'/list/{i}')
		else:
			response = requests.get(base_url)
		html = response.text
		html = html.split('class="item-name">\r\n\t\t\t\t\t\t\t\t\t\t')
		for div in html[1:]:
			corr_div = div.split('\r\n\t')
			name = corr_div[0]
			if name.lower() not in artists:
				artists.append(name.lower())


scrape_artists()
'''
'offset',  
		'xxxtentacion',
		'the weeknd', 
		'21 savage', 
		'gunna',
		'gucci mane',
		'drake', 
		'quavo', 
		'schoolboy q', 
		'nicki minaj', 
		'sza',
		'lil uzi vert',
		'playboi carti',
		'travis scott',
		'lil baby',
	    'dababy',
	    'pop smoke',
		'meek mill',
		'future',
		'trippie redd', 
		'young thug',
		'wiz khalifa',
		'swae lee',
		'nba youngboy',
		'juice wrld',
		'ynw melly',
		'lil mosey',
		'shoreline mafia',
		'don toliver',
		'jack harlow',
		'lil yachty',
		'asap rocky',
		'2 chainz',
		'young dolph',
		'nav',
		'lil tjay',
		'suicideboys',
		'ramirez',
		'lil peep',
		'night lovell',
		'fat nick',
		'shakewell',
		'roddy ricch',
		'frank ocean',
		'smokepurpp',
		'lil pump',
		'post malone',
		'kendrick lamar',
		'j cole',
		'j. cole',
		'fivio foreign',
		'skepta',
		'kanye west',
		'eminem',
		'50 cent',
		'lil durk',
		'dj khaled',
		'lil wayne',
		'rick ross',
		'mac miller',
		'kid cudi',
		'pooh shiesty',
		'moneybagg yo',
		'baby keem',
		'polo g',
		'lil tecca',
		'internet money',
		'blueface',
		'joyner lucas',
		'tee grizzley',
		'mario judah',
		'ice cube',
		'dr dre',
'''