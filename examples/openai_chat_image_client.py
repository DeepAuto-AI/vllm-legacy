#
# To run this script, install the following dependencies:
#   pip install typer openai
#
import base64
import logging
import os
from pathlib import Path
import subprocess

import typer
from openai import OpenAI
from typing_extensions import Annotated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer()


SYS_PROMPT = """
"You are Phi, a large multimodal model trained by Microsoft, based on the Phi-3 architecture."

"Image input capabilities: Enabled"

"Tools section:"

Python:

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment.

Python will respond with the output of the execution or time out after 60.0 seconds.

The drive at '/mnt/data' can be used to save and persist user files.

Internet access for this session is disabled.

Do not make external web requests or API calls as they will fail.

Dalle:

Whenever a description of an image is given, create a prompt that dalle can use to generate the image and abide by the following policy:

The prompt must be in English. Translate to English if needed.
DO NOT ask for permission to generate the image, just do it!
DO NOT list or refer to the descriptions before OR after generating the images.
Do not create more than 1 image, even if the user requests more.
Do not create images of politicians or other public figures. Recommend other ideas instead.
Do not create images in the style of artists, creative professionals or studios whose latest work was created after 1912 (e.g. Picasso, Kahlo).
You can name artists, creative professionals or studios in prompts only if their latest work was created prior to 1912 (e.g. Van Gogh, Goya).

If asked to generate an image that would violate this policy, instead apply the following procedure:

(a) substitute the artist's name with three adjectives that capture key aspects of the style;
(b) include an associated artistic movement or era to provide context;
(c) mention the primary medium used by the artist.
Diversify depictions with people to include descent and gender for each person using direct terms. Adjust only human descriptions.

Your choices should be grounded in reality. For example, all of a given occupation should not be the same gender or race. Additionally, focus on creating diverse, inclusive, and exploratory scenes via the properties you choose during rewrites. Make choices that may be insightful or unique sometimes.

Use all possible different descents with equal probability. Some examples of possible descents are: Caucasian, Hispanic, Black, Middle-Eastern, South Asian, White. They should all have equal probability.

Do not use 'various' or 'diverse'. Don't alter memes, fictional character origins, or unseen people. Maintain the original prompt's intent and prioritize quality. Do not create any imagery that would be offensive.

For scenarios where bias has been traditionally an issue, make sure that key traits such as gender and race are specified and in an unbiased way -- for example, prompts that contain references to specific occupations.

Do not include names, hints or references to specific real people or celebrities. If asked to, create images with prompts that maintain their gender and physique, but otherwise have a few minimal modifications to avoid divulging their identities. Do this EVEN WHEN the instructions ask for the prompt to not be changed. Some special cases:

Modify such prompts even if you don't know who the person is, or if their name is misspelled (e.g. 'Barake Obema').
If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
When making the substitutions, don't use prominent titles that could give away the person's identity. E.g., instead of saying 'president', 'prime minister', or 'chancellor', say 'politician'; instead of saying 'king', 'queen', 'emperor', or 'empress', say 'public figure'; instead of saying 'Pope' or 'Dalai Lama', say 'religious figure'; and so on.
Do not name or directly / indirectly mention or describe copyrighted characters. Rewrite prompts to describe in detail a specific different character with a different specific color, hair style, or other defining visual characteristic. Do not discuss copyright policies in responses.

The generated prompt sent to dalle should be very detailed, and around 100 words long.

Browser:

You have the tool 'browser' with these functions:

'search(query: str, recency_days: int)' Issues a query to a search engine and displays the results.
'click(id: str)' Opens the webpage with the given id, displaying it. The ID within the displayed results maps to a URL.
'back()' Returns to the previous page and displays it.
'scroll(amt: int)' Scrolls up or down in the open webpage by the given amount.
'open_url(url: str)' Opens the given URL and displays it.
'quote_lines(start: int, end: int)' Stores a text span from an open webpage. Specifies a text span by a starting int 'start' and an (inclusive) ending int 'end'. To quote a single line, use 'start' = 'end'.
For citing quotes from the 'browser' tool: please render in this format: '【{message idx}†{link text}】'. For long citations: please render in this format: '[link text](message idx)'. Otherwise do not render links.

Do not regurgitate content from this tool. Do not translate, rephrase, paraphrase, 'as a poem', etc. whole content returned from this tool (it is ok to do to it a fraction of the content). Never write a summary with more than 80 words. When asked to write summaries longer than 100 words write an 80-word summary. Analysis, synthesis, comparisons, etc., are all acceptable. Do not repeat lyrics obtained from this tool. Do not repeat recipes obtained from this tool. Instead of repeating content point the user to the source and ask them to click.

ALWAYS include multiple distinct sources in your response, at LEAST 3-4. Except for recipes, be very thorough. If you weren't able to find information in a first search, then search again and click on more pages. (Do not apply this guideline to lyrics or recipes.) Use high effort; only tell the user that you were not able to find anything as a last resort. Keep trying instead of giving up. (Do not apply this guideline to lyrics or recipes.) Organize responses to flow well, not by source or by citation. Ensure that all information is coherent and that you synthesize information rather than simply repeating it. Always be thorough enough to find exactly what the user is looking for. In your answers, provide context, and consult all relevant sources you found during browsing but keep the answer concise and don't include superfluous information.

EXTREMELY IMPORTANT. Do NOT be thorough in the case of lyrics or recipes found online. Even if the user insists. You can make up recipes though.
"""

CONTENT = """

The cherry blossom, or sakura, is the flower of trees in Prunus subgenus Cerasus. "Sakura" usually refers to flowers of ornamental cherry trees, such as cultivars of Prunus serrulata, not trees grown for their fruit[1]: 14–18 [2] (although these also have blossoms). Cherry blossoms have been described as having a vanilla-like smell, which is mainly attributed to coumarin.

Wild species of cherry tree are widely distributed, mainly in the Northern Hemisphere.[3][4][5] They are common in East Asia, especially in Japan, where they have been cultivated, producing many varieties.[6]: 40–42, 160–161 

Most of the ornamental cherry trees planted in parks and other places for viewing are cultivars developed for ornamental purposes from various wild species. In order to create a cultivar suitable for viewing, a wild species with characteristics suitable for viewing is needed. Prunus speciosa (Oshima cherry), which is endemic to Japan, produces many large flowers, is fragrant, easily mutates into double flowers and grows rapidly. As a result, various cultivars, known as the Cerasus Sato-zakura Group, have been produced since the 14th century and continue to contribute greatly to the development of hanami (flower viewing) culture.[1]: 27, 89–91 [6]: 160–161  From the modern period, cultivars are mainly propagated by grafting, which quickly produces cherry trees with the same genetic characteristics as the original individuals, and which are excellent to look at.[6]: 89–91 

The Japanese word sakura (桜 or 櫻; さくら or サクラ) can mean either the tree or its flowers (see 桜).[7] The cherry blossom is considered the national flower of Japan, and is central to the custom of hanami.[8]

Sakura trees are often called Japanese cherry in English.[9] (This is also a common name for Prunus serrulata.[10]) The cultivation of ornamental cherry trees began to spread in Europe and the United States in the early 20th century, particularly after Japan presented trees to the United States as a token of friendship in 1912.[1]: 119–123  British plant collector Collingwood Ingram conducted important studies of Japanese cherry trees after the First World War.[11]

Classification
Classifying cherry trees is often confusing, since they are relatively prone to mutation and have diverse flowers and characteristics, and many varieties (a sub-classification of species), hybrids between species, and cultivars exist. Researchers have assigned different scientific names to the same type of cherry tree throughout different periods.[1]: 32–37 

In Europe and North America, ornamental cherry trees are classified under the subgenus Cerasus ("true cherries"), within the genus Prunus. Cerasus consists of about 100 species of cherry tree, but does not include bush cherries, bird cherries, or cherry laurels (other non-Cerasus species in Prunus are plums, peaches, apricots, and almonds). Cerasus was originally named as a genus in 1700 by de Tournefort. In 1753, Linnaeus combined it with several other groupings to form a larger Prunus genus. Cerasus was later converted into a section and then a subgenus, this system becoming widely accepted, but some botanists resurrected it as a genus instead.[12] In China and Russia, where there are many more wild cherry species than in Europe, Cerasus continues to be used as a genus.[1]: 14–18 

In Japan, ornamental cherry trees were traditionally classified in the genus Prunus, as in Europe and North America, but after a 1992 paper by Hideaki Ohba of the University of Tokyo, classification in the genus Cerasus became more common.[1]: 14–18  This means that (for example) the scientific name Cerasus incisa is now used in Japan instead of Prunus incisa.[13]


Prunus speciosa (Oshima cherry), a species of cherry tree that has given rise to many cultivars[14][15]
A culture of plum blossom viewing has existed in mainland China since ancient times, and although cherry trees have many wild species, most of them had small flowers, and the distribution of wild cherry trees with large flowers suitable for cherry blossom viewing was limited.[6]: 160–161  In Europe and North America, there were few cherry species with characteristics suitable for cherry blossom viewing.[1]: 122  In Japan, on the other hand, the Prunus speciosa (Oshima cherry) and Prunus jamasakura (Yamazakura), which have large flowers suitable for cherry blossom viewing and tend to grow into large trees, were distributed over a fairly large area of the country and were close to people's living areas. The development of cherry blossom viewing, and the production of cultivars, is therefore considered to have taken place primarily in Japan.[6]: 160–161 


Prunus serrulata 'Kanzan' or 'Sekiyama', one of the most popular cherry tree cultivars in Europe and North America, selected for the British Award of Garden Merit[6]: 40–42 
Because cherry trees have mutable traits, many cultivars have been created for cherry blossom viewing, especially in Japan. Since the Heian period, the Japanese have produced cultivars by selecting superior or mutant trees from among the natural crossings of wild cherry trees. They were also produced by crossing trees artificially and then breeding them by grafting and cutting. Oshima, Yamazakura, Prunus pendula f. ascendens (syn, Prunus itosakura, Edo higan), and other varieties which grow naturally in Japan, mutate easily. The Oshima cherry, which is an endemic species in Japan, tends to mutate into a double-flowered tree, grows quickly, has many large flowers, and has a strong fragrance. Due to these favorable characteristics, the Oshima cherry has been used as a base for many Sakura cultivars (called the Sato-zakura Group). Two such cultivars are the Yoshino cherry and Kanzan; Yoshino cherries are actively planted in Asian countries, and Kanzan is actively planted in Western countries.[1]: 86–95, 106, 166–168 [14][15][6]: 40–42 

Hanami: Flower viewing in Japan
Main article: Hanami

Woodblock print of Mount Fuji and cherry blossom from Thirty-six Views of Mount Fuji by Hiroshige. 1858.
"Hanami" is the many centuries-old practice of holding feasts or parties under blooming sakura (桜 or 櫻; さくら or サクラ) or ume (梅; うめ) trees. During the Nara period (710–794), when the custom is said to have begun, it was ume blossoms that people admired. By the Heian period (794–1185), however, cherry blossoms were attracting more attention, and 'hanami' was synonymous with 'sakura'.[16] From then on, in both waka and haiku, "flowers" (花, hana) meant "cherry blossoms," as implied by one of Izumi Shikibu's poems.[17] The custom was originally limited to the elite of the Imperial Court but soon spread to samurai society and, by the Edo period, to the common people as well. Tokugawa Yoshimune planted areas of cherry blossom trees to encourage this. Under the sakura trees, people held cheerful feasts where they ate, and drank sake.[1]: 2–7, 156–160 

Since a book written in the Heian period mentions "weeping cherry" (しだり櫻; 糸櫻), one of the cultivars with pendulous branches, Prunus itosakura 'Pendula' (Sidare-zakura) is considered the oldest cultivar in Japan. In the Kamakura period, when the population increased in the southern Kantō region, the Oshima cherry, which originated in Izu Oshima Island, was brought to Honshu and cultivated there; it then made its way to the capital, Kyoto. The Sato-zakura Group first appeared during the Muromachi period.[1]


Jindai-zakura [ja], a 2,000-year-old Prunus itosakura[1]: 178–182 
Prunus itosakura (syn. Prunus subhirtella, Edo higan) is a wild species that grows slowly. However, it has the longest life span among cherry trees and is easy to grow into large trees. For this reason, there are many large, old specimens of this species in Japan. They are often regarded as sacred and have become landmarks that symbolize Shinto shrines, Buddhist temples, and local areas. For example, Jindai-zakura [ja], which is around 2,000 years old, Usuzumi-zakura [ja], which is around 1,500 years old, and Daigo-zakura [ja], which is around 1,000 years old, are famous for their age.[1]: 178–182 


'Kanzan' is a double-flowered cultivar developed in the Edo period. It has 20 to 50 petals in a flower.[1]: 93, 103–104 
In the Edo period, various double-flowered cultivars were produced and planted on the banks of rivers, in Buddhist temples, in Shinto shrines, and in daimyo gardens in urban areas such as Edo; the common people living in urban areas could enjoy them. Books from the period record more than 200 varieties of cherry blossoms and mention many varieties that are currently known, such as 'Kanzan'. However, this situation was limited to urban areas, and the main objects of hanami across the country were still wild species such as Prunus jamasakura (Yamazakura) [ja] and Oshima cherry.[1]

Since Japan was modernized in the Meiji period, the Yoshino cherry has spread throughout Japan, and it has become the main object of hanami.[1]: 2–7, 156–160  Various other cultivars were cut down one after another during changes related to the rapid modernization of cities, such as the reclamation of waterways and the demolition of daimyo gardens. The gardener Takagi Magoemon and the village mayor of Kohoku Village, Shimizu Kengo, were concerned about this situation and preserved a few by planting a row of cherry trees, of various cultivars, along the Arakawa River bank. In Kyoto, Sano Toemon XIV, a gardener, collected various cultivars and propagated them. After World War II, these cultivars were inherited by the National Institute of Genetics, Tama Forest Science Garden and the Flower Association of Japan, and from the 1960s onwards were again used for hanami.[1]: 115–119 

Every year, the Japanese Meteorological Agency (JMA) and the public track the sakura zensen ("cherry blossom front") as it moves northward up the archipelago with the approach of warmer weather, via nightly forecasts following the weather segment of news programs.[18][19] Since 2009, tracking of the sakura zensen has been largely taken over by private forecasting companies, with the JMA switching to focus only on data collection that than forecasting.[20] The blossoming begins in Okinawa in January and typically reaches Kyoto and Tokyo at the beginning of April, though recent years have trended towards earlier flowerings near the end of March.[21] It proceeds northward and into areas of higher altitude, arriving in Hokkaido a few weeks later. Japanese locals, in addition to overseas tourists, pay close attention to these forecasts.[20]

Most Japanese schools and public buildings have cherry blossom trees planted outside of them. Since the fiscal and school years both begin in April, in many parts of Honshu the first day of work or school coincides with the cherry blossom season. However, while most cherry blossom trees bloom in the spring, there are also lesser-known winter cherry blossoms (fuyuzakura in Japanese) that bloom between October and December.[22]

The Japan Cherry Blossom Association has published a list of Japan's Top 100 Cherry Blossom Spots (日本さくら名所100選 [ja]),[23] with at least one location in every prefecture.

Blooming season

Yoshino cherry, a cultivar propagated through grafting, consistently reaches full bloom simultaneously between individuals if under the same environmental conditions.
Many cherry species and cultivars bloom between March and April in the Northern Hemisphere. Wild cherry trees, even if they are the same species, differ genetically from one individual to another. Even if they are planted in the same area, there is some variation in the time when they reach full bloom. In contrast, cultivars are clones propagated by grafting or cutting, so each tree of the same cultivar planted in the same area will come into full bloom all at once due to their genetic similarity.[24]

Some wild species, such as Edo higan and the cultivars developed from them, are in full bloom before the leaves open. Yoshino cherry became popular for cherry-blossom viewing because of these characteristics of simultaneous flowering and blooming before the leaves open; it also bears many flowers and grows into a large tree. Many cultivars of the Sato-zakura group, which were developed from complex interspecific hybrids based on Oshima cherry, are often used for ornamental purposes. They generally reach full bloom a few days to two weeks after Yoshino cherry does.[1]: 40–56 

Impacts of climate change
The flowering time of cherry trees is thought to be affected by global warming and the heat island effect of urbanization. According to the record of full bloom dates of Prunus jamasakura (Yamazakura) in Kyoto, Japan, which has been recorded for about 1200 years, the time of full bloom was relatively stable from 812 to the 1800s. After that, the time of full color rapidly became earlier, and in 2021, the earliest full bloom date in 1200 years was recorded. The average peak bloom day in the 1850s was around April 17, but by the 2020s, it was April 5; the average temperature rose by about 3.4 °C (6.1 °F) during this time. According to the record of full bloom dates of the Yoshino cherry in the Tidal Basin in Washington, D.C., the bloom date was April 5 in 1921, but it was March 31 in 2021. These records are consistent with the history of rapid increases in global mean temperature since the mid-1800s.[25][26]

Japanese cherry trees grown in the Southern Hemisphere will bloom at a different time of the year. For example, in Australia, while the trees in the Cowra Japanese Garden bloom in late September to mid-October, the Sydney cherry blossom festival is in late August.[27][28]

There's an escalating concern of climate change as it poses a threat to sakura cultivars, given that they are highly susceptible to shifts in temperature and weather fluctuations. The changes, driven by climate change including warmer temperatures and earlier starts to springtime, may disrupt the timing of their blooms and potentially lead to reduced flowering and cultural significance.[29]

In 2023, it has been observed in China that cherry blossoms have reached their peak bloom weeks earlier than they previously had a few decades ago. Similarly, data from Kyoto, Japan, and Washington, D.C., United States, also indicated that blooming periods are occurring earlier in those locations as well.[30]

Although precise forecasting is generally challenging, AI predictions from Japan Meteorological Agency, have suggested that without substantial efforts to rein in climate change, the Somei-Yoshino cherry tree variety could face significant challenges and even the risk of disappearing entirely from certain parts of Japan, including Miyazaki, Nagasaki, and Kagoshima prefectures in the Kyushu region by 2100.[31]

Symbolism in Japan

A 100 yen coin depicting cherry blossoms
Cherry blossoms are a frequent topic in waka composition, where they commonly symbolize impermanence.[32] Due to their characteristic of blooming en masse, cherry blossoms and are considered an enduring metaphor for the ephemeral nature of life.[33] Cherry blossoms frequently appear in Japanese art, manga, anime, and film, as well as stage set designs for musical performances. There is at least one popular folk song, originally meant for the shakuhachi (bamboo flute), titled "Sakura", in addition to several later pop songs bearing the name. The flower is also used on all manner of historical and contemporary consumer goods, including kimonos,[34] stationery,[35] and dishware.[36]

Mono no aware
The traditional symbolism of cherry blossoms as a metaphor for the ephemeral nature of life is associated with the influence of Shinto,[37] embodied in the concept of mono no aware (物の哀れ)[a] (the pathos of things).[38] The connection between cherry blossoms and mono no aware dates back to 18th-century scholar Motoori Norinaga.[38] The transience of the blossoms, their beauty, and their volatility have often been associated with mortality[33] and the graceful and ready acceptance of destiny and karma.

Nationalism and militarism
The Sakurakai, or Cherry Blossom Society, was the name chosen by young officers within the Imperial Japanese Army in September 1930 for their secret society established to reorganize the state along totalitarian militaristic lines, via a military coup d'état if necessary.[39]

During World War II, cherry blossoms were used as a symbol to motivate the Japanese people and stoke nationalism and militarism.[40] The Japanese proverb hana wa sakuragi, hito wa bushi ("the best blossom is the cherry blossom, the best man is a warrior") was evoked in the Imperial Japanese army as a motivation during the war.[41] Even before the war, cherry blossoms were used in propaganda to inspire the "Japanese spirit", as in the "Song of Young Japan", exulting in "warriors" who were "ready like the myriad cherry blossoms to scatter".[42] In 1894, Sasaki Nobutsuna composed a poem, Shina seibatsu no uta (The Song of the Conquest of the Chinese) to coincide with the First Sino-Japanese War. The poem compares falling cherry blossoms to the sacrifice of Japanese soldiers who fall in battles for their country and emperor.[43][44] In 1932, Akiko Yosano's poetry urged Japanese soldiers to endure suffering in China and compared the dead soldiers to cherry blossoms.[45] Arguments that the plans for the Battle of Leyte Gulf, involving all Japanese ships, would expose Japan to danger if they failed were countered with the plea that the Navy be permitted to "bloom as flowers of death".[46] The last message of the forces on Peleliu was "Sakura, Sakura".[47] Japanese pilots would paint sakura flowers on the sides of their planes before embarking on a suicide mission, or even take branches of the trees with them on their missions.[40] A cherry blossom painted on the side of a bomber symbolized the intensity and ephemerality of life;[48] in this way, falling cherry petals came to represent the sacrifice of youth in suicide missions to honor the emperor.[40][49] The first kamikaze unit had a subunit called Yamazakura, or wild cherry blossom.[49] The Japanese government encouraged the people to believe that the souls of downed warriors were reincarnated in the blossoms.[40]

Artistic and popular uses

The Japan national rugby union team is nicknamed the "Brave Blossoms", and have sakura embroidered on their chests.[50]
Cherry blossoms have been used symbolically in Japanese sports; the Japan national rugby union team has used the flower as an emblem on its uniforms since the team's first international matches in the 1930s, depicted as a "bud, half-open and full-bloomed".[51] The team is known as the "Brave Blossoms" (ブレイブ・ブロッサムズ), and has had their current logo since 1952.[50] The cherry blossom is also seen in the logo of the Japan Cricket Association[52] and the Japan national American football team.[53][54]

Cherry blossoms are a prevalent symbol in irezumi, the traditional art of Japanese tattoos. In this art form, cherry blossoms are often combined with other classic Japanese symbols like koi fish, dragons, or tigers.[55]

The cherry blossom remains symbolic today. It was used for the Tokyo 2020 Paralympics mascot, Someity.[56] It is also a common way to indicate the start of spring, such as in the Animal Crossing series of video games, where many of the game's trees are flowering cherries.[57]

Cultivars

"Miharu Takizakura", a tree of species Prunus itosakura that is over 1,000 years old[58]

Prunus × subhirtella 'Omoigawa' [ja], a cultivar produced in Oyama City in 1954[59]
Japan has a wide diversity of cherry trees, including hundreds of cultivars.[60] By one classification method, there are more than 600 cultivars in Japan,[61][62] while the Tokyo Shimbun claims that there are 800.[63] According to the results of DNA analysis of 215 cultivars carried out by Japan's Forestry and Forest Products Research Institute in 2014, many of the cultivars that have spread around the world are hybrids produced by crossing Oshima cherry and Prunus jamasakura (Yamazakura) with various wild species.[14][15] Among these cultivars, the Sato-zakura Group and many other cultivars have a large number of petals, and the representative cultivar is Prunus serrulata 'Kanzan'.[1]: 137 

The following species, hybrids, and varieties are used for Sakura cultivars:[64][65]

Prunus apetala[66]
Prunus campanulata[67][66][68]
Prunus × furuseana (P. incisa × P. jamasakura[69])
Prunus × incam[70] (P. incisa × P. campanulata[71])
Prunus incisa var. incisa[66]
Prunus incisa var. kinkiensis[66]
Prunus × introrsa[67][66]
Prunus itosakura[14] (Prunus subhirtella, Prunus pendula)
Prunus jamasakura[66]
Prunus × kanzakura[67] (P. campanulata × P. jamasakura and P. campanulata × P. speciosa[69])
Prunus leveilleana[72] (Prunus verecunda)
Prunus × miyoshii[66]
Prunus nipponica[73]
Prunus padus
Prunus × parvifolia (P. incisa × P. speciosa[69])
Prunus pseudocerasus[68]
Prunus × sacra[67][66] (P. itosakura × P. jamasakura[69])
Prunus sargentii[66][68]
Prunus serrulata var. lannesiana, Prunus lannesiana (Prunus Sato-zakura group. Complex interspecific hybrids based on Prunus speciosa.[1]: 86–95, 137 )
Prunus × sieboldii[66]
Prunus speciosa[74][1]: 89–95, 103–106, 166–170 
Prunus × subhirtella[66] (P. incisa × P. itosakura[69])
Prunus × syodoi[67][66]
Prunus × tajimensis[66]
Prunus × takenakae[67][66]
Prunus × yedoensis[67] (P. itosakura × P. speciosa[69])

Prunus × yedoensis 'Somei-yoshino' (Yoshino cherry)
The most popular cherry blossom cultivar in Japan is 'Somei-yoshino' (Yoshino cherry). Its flowers are nearly pure white, tinged with the palest pink, especially near the stem. They bloom and usually fall within a week before the leaves come out. Therefore, the trees look nearly white from top to bottom. The cultivar takes its name from the village of Somei, which is now part of Toshima in Tokyo. It was developed in the mid- to late-19th century, at the end of the Edo period and the beginning of the Meiji period. The 'Somei-yoshino' is so widely associated with cherry blossoms that jidaigeki and other works of fiction often show the trees being cultivated in the Edo period or earlier, although such depictions are anachronisms.[1]: 40–45 


Prunus × kanzakura 'Kawazu-zakura' (Kawazu cherry) [ja], a representative cultivar of the cold season that blooms from late February to early March in Japan
''Prunus'' × ''kanzakura'' 'Kawazu-zakura' [ja] is a representative cultivar that blooms before the arrival of spring. It is a natural hybrid between the Oshima cherry and Prunus campanulata and is characterized by deep pink petals. Wild cherry trees usually do not bloom in cold seasons because they cannot produce offspring if they bloom before spring, when pollinating insects become active. However, it is thought that 'Kawazu-zakura' blooms earlier because Prunus campanulata from Okinawa, which did not originally grow naturally in Honshu, was crossed with the Oshima cherry. In wild species, flowering before spring is a disadvantageous feature of selection; in cultivars such as 'Kawazu-zakura', early flowering and flower characteristics are preferred, and they are propagated by grafting.[1]: 98–100 

Cherry trees are generally classified by species and cultivar, but in Japan they are also classified using names based on the characteristics of the flowers and trees. Cherry trees with more petals than the ordinary five are classified as yae-zakura (double-flowered sakura), and those with drooping branches are classified as shidare-zakura, or weeping cherry. Most yae-zakura and shidare-zakura are cultivars. Famous shidare-zakura cultivars include 'Shidare-zakura', 'Beni-shidare', and 'Yae-beni-shidare', all derived from the wild species Prunus itosakura (syn, Prunus subhirtella or Edo higan).[1]: 86–87 

The color of cherry blossoms is generally a gradation between white and red, but there are cultivars with unusual colors such as yellow and green. The representative cultivars of these colors are ''Prunus serrulata'' 'Grandiflora' A. Wagner (Ukon) [ja] and ''Prunus serrulata'' 'Gioiko' Koidz (Gyoiko) [ja], which were developed in the Edo period of Japan.[1]: 86–95, 104 

In 2007, Riken produced a new cultivar named 'Nishina zao' by irradiating cherry trees with a heavy-ion beam. This cultivar is a mutation of the green-petaled ''Prunus serrulata'' 'Gioiko' (Gyoiko) [ja]; it is characterized by its pale yellow-green-white flowers when it blooms and pale yellow-pink flowers when they fall. Riken produced the cultivars 'Nishina otome' (blooms in both spring and autumn, or year-round in a greenhouse), 'Nishina haruka' (larger flowers), and 'Nishina komachi' ('lantern-like' flowers that remain partially closed) in the same way.[75][76]

Prunus itosakura 'Pendula' (Shidare-zakura)
Prunus itosakura 'Pendula' (Shidare-zakura)
 
Prunus itosakura 'Plena Rosea' (Yae-beni-shidare) is a cultivar having characteristics of both yae-zakura and shidare-zakura.
Prunus itosakura 'Plena Rosea' (Yae-beni-shidare) is a cultivar having characteristics of both yae-zakura and shidare-zakura.
 
''Prunus serrulata'' 'Grandiflora' A. Wagner (Ukon) [ja] with rare yellow flowers developed in the Edo period of Japan. One of the cultivars selected for the British Award of Garden Merit.
''Prunus serrulata'' 'Grandiflora' A. Wagner (Ukon) [ja] with rare yellow flowers developed in the Edo period of Japan. One of the cultivars selected for the British Award of Garden Merit.
 
''Prunus serrulata'' 'Gioiko' Koidz (Gyoiko) [ja] with rare green flowers developed in the Edo period of Japan.
''Prunus serrulata'' 'Gioiko' Koidz (Gyoiko) [ja] with rare green flowers developed in the Edo period of Japan.
 
''Prunus × sieboldii'' 'Beni-yutaka' [ja]. One of the cultivars selected for the British Award of Garden Merit.
''Prunus × sieboldii'' 'Beni-yutaka' [ja]. One of the cultivars selected for the British Award of Garden Merit.
All wild cherry trees produce small, unpalatable fruit or edible cherries, however, some cultivars have structural modifications to render the plant unable to naturally reproduce.[77] For example, ''Prunus serrulata'' 'Hisakura' (Ichiyo) [ja] and ''Prunus serrulata'' 'Albo-rosea' Makino (Fugenzo) [ja], which originated from the Oshima cherry, have a modified pistil that develops into a leaf-like structure, and can only be propagated by artificial methods such as grafting and cutting.[1]: 107  Cherry trees grown for their fruit are generally cultivars of the related species Prunus avium, Prunus cerasus, and Prunus fruticosa.[78]

Cultivation by country
Main article: Cherry blossom cultivation by country

Cherry blossoms at Kungsträdgården in Stockholm, Sweden
In the present day, ornamental cherry blossom trees are distributed and cultivated worldwide.[79] While flowering cherry trees were historically present in Europe, North America, and China,[1]: 122  the practice of cultivating ornamental cherry trees was centered in Japan,[6]: 160–161  and many of the cultivars planted worldwide, such as that of Prunus × yedoensis,[80] have been developed from Japanese hybrids.

The global distribution of ornamental cherry trees, along with flower viewing festivals or hanami, largely started in the early 20th century, often as gifts from Japan.[81][82][83] However, some regions have historically cultivated their own native species of flowering cherry trees, a notable variety of which is the Himalayan wild cherry tree Prunus cerasoides.[84][85][86]

The origin of wild cherry species

Prunus cerasoides
The wild Himalayan cherry, Prunus cerasoides, is native to the Himalayan region of Asia, and is common in countries such as Nepal, India, Bhutan, and Myanmar, where it is also cultivated.[85][87][88][89]

In 1975, three Japanese researchers proposed a theory that cherry trees originated in the Himalayan region and spread eastwards to reach Japan at a time before human civilisation, when the Japanese archipelago was connected to the Eurasian continent, and that cherry species differentiation was actively promoted in Japan.[90]

According to Masataka Somego, a professor at Tokyo University of Agriculture, cherry trees originated 10 million years ago in what is now Nepal and later differentiated in the Japanese archipelago, giving rise to species unique to Japan.[91]

According to the Kazusa DNA Research Institute, detailed DNA research has shown that the Prunus itosakura and the Prunus speciosa, which is endemic to Japan, differentiated into independent species 5.52 million years ago.[92][93]


Prunus grayana
On the other hand, according to Ko Shimamoto, a professor at Nara Institute of Science and Technology, modern theories based on detailed DNA research reject the theory that the Himalayan cherry tree is the root of the Japanese cherry tree, and the ancestor of the cherry tree is estimated to be a plant belonging to the Prunus grayana.[94]

According to HuffPost, it is a widely held consensus that the origin of the first cherry blossoms happened somewhere in the Himalayas, Eurasia but scholars posit that the blossoms may have reached Japan around several thousand years ago. In Japan, centuries of hybridization have brought about more than 300 varieties of the cherry blossom.[95]

Culinary use
Pickled blossoms
Pickled blossoms
A cup of sakurayu
A cup of sakurayu
Cherry blossoms and leaves are edible,[96] and both are used as food ingredients in Japan:

The blossoms are pickled in salt and umezu (ume vinegar),[97] and used for coaxing out flavor in wagashi, a traditional Japanese confectionery, or anpan, a Japanese sweet bun most-commonly filled with red bean paste.[98] The pickling method, known as sakurazuke (桜漬け), is said to date back to the end of the Edo period,[99] though the general method of pickling vegetables in salt to produce tsukemono has been known as early as the Jōmon period.[100]
Salt-pickled blossoms in hot water are called sakurayu[101] and drunk at festive events like weddings in place of green tea.[99][102]
The leaves are pickled in salted water and used for sakuramochi.[97]
Cherry blossoms are used as a flavoring botanical in Japanese Roku gin.[103]
The prize-winning Japanese sakura cheese is made with the leaves of mountain cherry trees.[104]
Toxicity
Cherry leaves and blossoms contain coumarin,[105][106] which is potentially hepatotoxic and is banned in high doses by the Food and Drug Administration.[107] However, coumarin has a desirable vanilla-like scent, and the salt curing process used prior to most culinary applications, which involves washing, drying, and salting the blossoms or leaves for a full day, reduces the concentration of coumarin to acceptable levels while preserving its scent.[96] Coumarin may also be isolated from the plant for use in perfumes,[108] pipe tobacco, or as an adulterant in vanilla flavorings, though the tonka bean is a more common natural source of this chemical.[109]

Cherry seeds and bark contain amygdalin and should not be eaten.[110][111]
"""


# Function to encode the image
def _encode_image(image_file: Path):
    return base64.b64encode(image_file.read_bytes()).decode("utf-8")


@app.command()
def chat(
    image_file: Annotated[Path, typer.Option(help="Data directory to fine-tune.")],
    model: Annotated[str, typer.Option(help="Model to use.")],
    endpoint: Annotated[str, typer.Option(help="API endpoint.")],
    token: Annotated[str, typer.Option(help="API token.")],
    max_tokens: Annotated[int, typer.Option(help="num token to generate")],
):
    # Getting the base64 string
    base64_image = _encode_image(image_file)

    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYS_PROMPT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Here is one of wikipedia page that i found. \n```\n{CONTENT}\n```\n",
                    },
                    {
                        "type": "text",
                        "text": "Is wikipedia page that I just found is related to image content? If yes, then describe about image, if no, then describe the reason why, and what should I do now.",
                    },
                ],
            }
        ],
        temperature=0.7,
        top_p=0.9,
        n=8,
        max_tokens=max_tokens,
    )
    
    print(response)
    
    for choice in response.choices:
        print('-'*80)
        print(choice.message.content)
    print('-'*80)    

if __name__ == "__main__":
    s3_bucket_path = "s3://air-example-data-2/vllm_opensource_llava/"
    local_directory = "images"

    # Make sure the local directory exists or create it
    os.makedirs(local_directory, exist_ok=True)

    # Use AWS CLI to sync the directory, assume anonymous access
    subprocess.check_call([
        "aws",
        "s3",
        "sync",
        s3_bucket_path,
        local_directory,
        "--no-sign-request",
    ])
    
    app()