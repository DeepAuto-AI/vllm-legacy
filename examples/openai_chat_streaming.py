import requests
import json, sys

# Define the API key (replace this with your actual API key)
API_KEY = 'sk-1234123'

# Get the input from the user
WIKI = r"""
'''South Korea''',{{efn|South Koreans use the name {{transliteration|ko|Hanguk}} ({{lang|ko-Hang-KR|한국}}, {{lang|ko-Hant-KR|韓國}}) when referring to South Korea or Korea as a whole. The literal translation of South Korea, {{transliteration|ko|Namhan}} ({{lang|ko-Hang-KR|남한}}, {{lang|ko-Hant-KR|南韓}}), is rarely used. North Koreans use {{transliteration|ko|Namchosŏn}} ({{lang|ko-Hang-KP|남조선}}, {{lang|ko-Hant-KP|南朝鮮}}) when referring to South Korea, derived from the North Korean name for Korea, {{transliteration|ko|Chosŏn}} ({{lang|ko-Hang-KP|조선}}, {{lang|ko-Hant-KP|朝鮮}}).}} officially the '''Republic of Korea''' ('''ROK'''),{{efn|{{Korean|hangul=대한민국|hanja=大韓民國|rr=Daehanminguk|lit="''Great [[Samhan|Han]] Republic''" or "''Great Korean Republic''"}}}} is a country in [[East Asia]]. It constitutes the southern part of the [[Korea|Korean Peninsula]] and borders [[North Korea]] along the [[Korean Demilitarized Zone]]; though it also claims the land border with [[China]] and [[Russia]].{{efn|South Korea's border with North Korea is a disputed border as both countries claim the entirety of the Korean Peninsula.}} The country's western border is formed by the [[Yellow Sea]], while its eastern border is defined by the [[Sea of Japan]]. South Korea claims to be the sole legitimate government of the entire peninsula and [[List of islands of South Korea|adjacent islands]]. It has a [[Demographics of South Korea|population]] of 51.96 million, of which roughly half live in the [[Seoul Capital Area]], the [[List of largest cities|ninth most populous metropolitan area in the world]]. Other major cities include [[Busan]], [[Daegu]] and [[Incheon]].

The Korean Peninsula was inhabited as early as the [[Lower Paleolithic]] period. Its [[Gojoseon|first kingdom]] was noted in Chinese records in the early 7th century BCE. Following the unification of the [[Three Kingdoms of Korea]] into [[Unified Silla|Silla]] and [[Balhae]] in the late 7th century, Korea was ruled by the [[Goryeo]] dynasty (918–1392) and the [[Joseon]] dynasty (1392–1897). The succeeding [[Korean Empire]] (1897–1910) was [[Japan–Korea Treaty of 1910|annexed in 1910]] into the [[Empire of Japan]]. [[Korea under Japanese rule|Japanese rule]] ended following [[Surrender of Japan|Japan's surrender]] in [[World War II]], after which Korea was [[Division of Korea|divided into two zones]]: a [[Soviet Civil Administration|northern]] zone occupied by the [[Soviet Union]], and a [[United States Army Military Government in Korea|southern]] zone [[Operation Blacklist Forty|occupied]] by the [[United States]]. After negotiations on [[Korean reunification|reunification]] failed, the southern zone became the Republic of Korea in August 1948, while the northern zone became the [[Communist state|communist]] [[North Korea|Democratic People's Republic of Korea]] the following month.

In 1950, a [[Operation Pokpung|North Korean invasion]] began the [[Korean War]], which ended in 1953 after extensive fighting involving the [[United States in the Korean War|American-led]] [[United Nations Command]] and the [[People's Volunteer Army]] from China with [[Soviet Union in the Korean War|Soviet assistance]]. The war [[Aftermath of the Korean War|left 3 million Koreans dead and the economy in ruins]]. The authoritarian [[First Republic of Korea]] led by [[Syngman Rhee]] was overthrown in the [[April Revolution]] of 1960. However, the [[Second Republic of Korea|Second Republic]] was incompetent as it could not control the revolutionary fervor. The [[May 16 coup]] of 1961 led by [[Park Chung Hee]] put an end to the Second Republic, signaling the start of the [[Third Republic of Korea|Third Republic]] in 1963. South Korea's devastated economy [[Miracle on the Han River|began to soar]] under Park's leadership, recording the [[List of countries by GDP (real) per capita growth rate|one of fastest rises in average GDP per capita]]. Despite lacking natural resources, the nation rapidly developed to become one of the [[Four Asian Tigers]] based on international trade and [[economic globalization]], integrating itself within the world economy with [[export-oriented industrialization]]. The [[Fourth Republic of Korea|Fourth Republic]] was established after the [[October Restoration]] of 1972, in which Park wielded absolute power. The [[Yushin Constitution]] declared that the president could suspend [[human rights|basic human rights]] and appoint a third of the parliament. Suppression of the opposition and human rights abuse by the government became more severe in this period. Even after [[Assassination of Park Chung Hee|Park's assassination]] in 1979, the authoritarian rule continued in the [[Fifth Republic of Korea|Fifth Republic]] led by [[Chun Doo-hwan]], which violently seized power by two coups and brutally suppressing the [[Gwangju Uprising]]. The [[June Democratic Struggle]] of 1987 [[1987 South Korean presidential election|ended authoritarian rule]], forming the current Sixth Republic. The country is now considered among the [[Democracy indices|most advanced democracies]] in Continental and East Asia.

South Korea maintains a [[unitary state|unitary]] [[presidential system|presidential republic]] under the [[Constitution of South Korea|1987 constitution]] with a unicameral legislature, the [[National Assembly]]. It is considered a [[regional power]] and a [[developed country]], with [[Economy of South Korea|its economy]] ranked as the world's [[List of countries by GDP (nominal)|twelfth-largest by nominal GDP]] and the [[List of countries by GDP (PPP)|fourteenth-largest by GDP (PPP)]]. Its citizens enjoy one of the world's [[List of sovereign states by Internet connection speeds|fastest Internet connection speeds]] and [[List of high-speed railway lines|densest high-speed railway networks]]. The country is the world's [[List of countries by exports|ninth-largest exporter]] and [[List of countries by imports|ninth-largest importer]]. Its [[Republic of Korea Armed Forces|armed forces]] are ranked as one of the world's strongest militaries, with the world's second-largest standing army by [[List of countries by number of military and paramilitary personnel|military and paramilitary personnel]]. In the 21st century, South Korea has been renowned for its globally influential pop culture, particularly in [[K-pop|music]], [[Korean drama|TV dramas]] and [[Cinema of South Korea|cinema]], a phenomenon referred to as the [[Korean Wave]]. It is a member of the [[OECD]]'s [[Development Assistance Committee]], the [[G20]], the [[Indo-Pacific Economic Framework|IPEF]], and the [[Paris Club]].
Korea}}

"""
input = f"Hello world! Here I found the wikipedia docuemnt. Please summarize it.\n\n=== Document Starts ===\n\n{WIKI}\n\n=== Document Ends ===\n\n"

# Prepare the request data to be sent to the GPT API
data = {
    'model': 'microsoft/Phi-3-vision-128k-instruct',
    'stream': True,
    'max_tokens': 512,
    'messages': [
        {
            'role': 'user',
            'content': input
        }
    ]
}

# Set the headers for the request
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + API_KEY
}

# Send the request to the OpenAI API and process each chunk of data as it arrives
response = requests.post('http://localhost:8888/v1/chat/completions', data=json.dumps(data), headers=headers, stream=True)

show_text = sys.argv[-1] == 'text'

if response.status_code == 200:
    prompt_throughput = 0
    throughputs = []
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            line = chunk.decode()
            if not show_text:
                print(line.strip())
            else:
                if line.startswith('data:') and not '[DONE]' in line:
                    line = line[5:]
                    data = json.loads(line.strip())
                    if 'choices' in data and len(data['choices']) > 0 and 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                        print(data['choices'][0]['delta']['content'], end='', flush=True)
                        if data['performance']['is_prompt']:
                            prompt_throughput = data['performance']['request_throughput']
                        else:
                            throughputs.append(data['performance']['request_throughput'])
                    else:
                        print(line.strip())
                else:
                    print()
                    print(line.strip())
    print('prompt throughput', prompt_throughput)
    print('decode throughput', sum(throughputs) / len(throughputs))
else:
    print("Request failed with status code: ", response.status_code)