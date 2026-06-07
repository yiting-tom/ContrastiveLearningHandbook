// Contrastive Learning — Evolution History deck
// Four-act narrative: "the field spent 5 years removing its crutches"
// Run: node evolution_deck.js  ->  contrastive_learning_evolution.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const PW = 13.33, PH = 7.5;
pres.author = "Yi-Ting Li";
pres.title = "對比學習的演化史";

// ---- palette ----
const NAVY = "0E1626";   // dark background
const NAVY2 = "182740";  // dark card
const NAVY3 = "23375C";  // dark card 2
const LIGHT = "F4F6FB";  // light background
const CARD = "FFFFFF";
const INK = "162039";    // dark text on light
const MUTE = "5C6B8A";   // muted text
const ICE = "C6D6F2";    // light muted text on dark
const WHITE = "FFFFFF";
const E1 = "F59E0B";     // Era 1 amber
const E2 = "06B6D4";     // Era 2 cyan
const E3 = "F43F5E";     // Era 3 rose (the heretical turn)
const E4 = "8B5CF6";     // Era 4 violet
const ACCENT = "22D3EE"; // bright cyan accent

const FACE = "PingFang TC";
const MONO = "Menlo";

const makeShadow = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 135, opacity: 0.18 });
const softShadow = () => ({ type: "outer", color: "0E1626", blur: 7, offset: 2, angle: 135, opacity: 0.12 });

// ---------- speaker notes (verbatim 逐字稿 + 🎬 舞台提示, one per slide) ----------
// Mirrors docs/TALK_SCRIPT.md so the presenter sees the spoken script in
// PowerPoint's Notes / Presenter View. Keys map 1:1 to slide order.
const N = (spoken, cue) => `${spoken}\n\n🎬 ${cue}`;
const NOTES = {
  cover: N(
    "來,我問大家一個問題喔。假設我給你一百萬張照片,裡面有貓、有狗、有車子、有房子⋯⋯但是,我一個標籤都不給你。沒有人告訴你哪張是貓、哪張是狗。那請問——模型,要怎麼自己學會分辨貓跟狗?聽起來有點玄對不對?好像在叫一個從來沒看過動物的小孩,自己去把貓跟狗分成兩堆。可是,接下來這二十幾分鐘,我要跟大家講一個故事——一個這個領域花了整整五年、一步一步教會機器做到這件事的故事。這個主題,叫做「對比學習」。",
    "站定中央,先不要急著翻頁。講到「一個標籤都不給你」時刻意停一拍,給聽眾消化。問句「要怎麼學會分辨貓跟狗?」之後停約 2 秒,讓懸念發酵,再進到標題。"),
  bigPicture: N(
    "在進細節之前,我想先給你一副「眼鏡」,讓你戴著它看完整場。這整個領域,五年來其實只在做一件事——不斷地「拿掉拐杖」。什麼意思?一開始,模型走路要靠很多外部的拐杖才站得起來;然後研究者就一根一根把拐杖抽掉,逼它自己站。我們會看到四步:2018 年,第一根拐杖被拿掉——人工標籤,不用標籤也能學;2020 年,拿掉一個叫 memory bank 的大型記憶庫,改用更聰明的設計;2021 年,更猛,連「負樣本」都拿掉了——這在當時幾乎是不可思議的;然後 2021 年之後,連底層的網路架構都換掉,從 CNN 換成 Transformer。所以記住這副眼鏡:四步、四根拐杖。等一下不管看到哪個方法,你都問自己一句話——它,又拿掉了哪一根拐杖?",
    "翻頁。可以伸出手指一根一根比「1、2、3、4」對應四步。講到「連負樣本都拿掉」時提高語氣、露出一點驚訝表情。最後「它又拿掉哪根拐杖?」放慢,當作貫穿全場的口號。"),
  coreIntuition: N(
    "好,那「沒有標籤怎麼學」到底是什麼魔法?其實核心直覺超簡單,簡單到你會覺得「就這樣?」。今天我拿同一張貓的照片,做兩種不同的變化——一張裁切一下、一張調個顏色、翻個面。對人類來說,這還是同一隻貓;那我們就告訴模型:這兩張,你要把它們在特徵空間裡「拉近」。反過來,另外一張狗的照片,跟我的貓,就要「推開」。一直拉近、一直推開,模型慢慢就學會去抓那個「不管你怎麼裁、怎麼變,都不會變的東西」——也就是語意。這就是它不用標籤,卻能分貓狗的祕密。可是——這裡有個陷阱喔。如果今天叫你「把同一張圖的兩個版本拉近」,最偷懶的解是什麼?就是把全世界所有圖,通通對應到同一個點!那「拉近」永遠滿分嘛。這個災難,有個名字,叫做「坍塌」。記住這個詞——坍塌——它會像鬼一樣,纏著接下來整場報告。",
    "翻頁,指向投影片上的「拉近 / 推開」示意。講「就這樣?」時用半開玩笑的語氣。最後講到「坍塌」時壓低聲音、放慢,埋伏筆——這是全場的核心張力,要演出來。"),
  infonce: N(
    "剛剛那個「拉近、推開」,寫成數學就是這條式子,叫 InfoNCE。你不用會背,只要看懂三個位置:分子,是正樣本——同一張圖那一對,我們要它「大」;分母,是一堆負樣本——不同的圖,我們要它「小」。還有一個小小的 τ,叫溫度,負責調鬆緊。記憶點就一句:分子拉、分母推。為什麼我要先把這條式攤開給你看?因為接下來十四個方法,全部都是在玩這條式子——不是改寫它,就是想盡辦法逃出它。",
    "翻頁,直接用手指或雷射筆點分子、再點分母,實體指出來。語速可以稍快,這是方法頁不要拖。最後「逃出它」三個字當鉤子,微微停頓接下一頁。"),
  evoMap: N(
    "最後,給大家一張地圖,等一下我們就照著它走。你看,橫著排開是四個時代:2018 的拿掉標籤、2020 的大批次與記憶庫、2021 的告別負樣本、再到 Transformer 時代。中間這些箭頭很關鍵——它不是時間先後而已,而是「誰解決了誰的瓶頸」。前一個方法卡住了、出問題了,下一個方法才應運而生。所以這不是一堆零散的論文,而是一條有因有果的演化鏈。好,故事正式開始——讓我們回到 2018 年,第一根拐杖,標籤,要被拿掉了。",
    "翻頁,手沿著時間軸由左掃到右,再點幾個箭頭強調「因果」。最後一句「回到 2018」當作進入 ACT 1 的轉場,語氣上揚、製造起跑感,然後停頓交給下一頁。"),
  act1: N(
    "好,故事正式開始。把時間拉回到二零一八年。當時大家被一個很基本、但很煩人的問題卡住:我手上有一大堆圖片,可是完全沒有標籤——沒有人告訴我哪張是貓、哪張是狗。問題來了,模型要學什麼?你總得給它一個「任務」吧?可是任務從哪來?標籤就是我們最大的那根拐杖,現在這根拐杖被抽掉了。第一幕,我們就來看前人怎麼無中生有,硬是替模型生出一個任務。",
    "走到舞台中央,停頓一拍再開口。說到「沒有標籤」時攤手;說到「無中生有」時故意放慢、製造懸念,然後切下一頁。"),
  instance_discrimination: N(
    "第一個答案聰明到有點任性:既然沒有標籤,那我就規定——每一張圖,就是它自己的一類。一百萬張圖,就是一百萬類。模型的任務,就是把每張圖跟其他所有圖區分開來。但這裡有個工程難題:你不可能每一步都去比對全部一百萬張。所以架構上他們用了一個 memory bank,把每張圖的特徵先存起來,當作現成的負樣本庫;loss 則想用 softmax——白話講就是「在一堆候選裡挑出正確那一個」的機率——但候選有上百萬個、分母算不動,只好用 NCE 這個抽樣技巧去近似它(只抽一小撮負樣本來算,而不是全部)。記憶點就一個:它把每張圖當成一類。但它撞牆的地方也很經典——bank 裡存的特徵是好幾輪以前算的,模型一直在進步,bank 卻在吃回頭草,特徵會過時。這個「過時」,就是下一步要解決的痛。",
    "比出「一張圖 = 一類」的手勢。講到 memory bank 時指向圖中的 bank 方塊。「吃回頭草」這句加重語氣,帶點調侃。"),
  invariant_spread: N(
    "隔年就有人說:那我乾脆不要 bank 了。Invariant Spread 的想法很直接——同一張圖做兩種增強,這兩個視角要拉近,叫 invariant;不同的圖要推開,叫 spread。負樣本不從什麼 bank 拿,就用這一個 batch 裡的其他圖,當下算、當下用,絕對不會過時。架構上它就是兩條共享權重的分支——同一張圖的兩個視角各走一條,loss 則是把開場那條 InfoNCE 對稱地算兩次。跟前一個比,差別就在拿掉了 memory bank、改用 in-batch 負樣本(直接用這個 batch 裡的其他圖)。記憶點是:它其實就是 SimCLR 的直系祖先,核心配方在這裡已經成形了,只是當年沒有把規模拉到爆。所以你看,我們才剛拿掉標籤,馬上又順手把 memory bank 這根拐杖也丟了。",
    "用兩手比「拉近」與「推開」。講到「直系祖先」時停頓,眼神看向觀眾,埋伏筆。「丟了」收尾乾脆。"),
  act2: N(
    "好,進入第二幕,而這一幕是整個故事最精彩的分岔。前面我們確立了一件事:對比學習要 work,你得有大量、而且新鮮的負樣本去把表徵推開,不然就會坍塌成一坨。問題是——這麼多負樣本,到底要從哪裡生出來?二零二零年,兩派人馬給了兩個完全不同的答案,直接吵成一場世紀對決。一派說「我用 queue 存起來」,另一派說「我 batch 開到爆就好」。我們先看第一派。",
    "語速稍微加快,點出「最精彩的分岔」。講到「兩派」時左右手各指一邊,做出對峙的架勢。"),
  moco_v1: N(
    "第一派是何愷明團隊的 MoCo。他們的洞見是:負樣本不一定要在這一個 batch 裡,我可以拿一條 FIFO 佇列——先進先出,新特徵推進去、最舊的被擠掉——把過去好幾個 batch 的特徵排隊存起來當負樣本。這樣負樣本的數量,就跟你的 batch size 完全脫鉤了——batch 很小,我照樣有上萬個負樣本。可是等等,這不就回到 Instance Discrimination「特徵會過時」的老問題嗎?這就是他們最漂亮的一招:動量編碼器。負樣本那一邊的編碼器不用梯度更新,而是用動量(也就是 EMA)——一步只吸收一點點主編碼器的更新、慢慢平滑地跟上,所以 queue 裡的特徵雖然舊,卻一致、不會亂跳。loss 本身還是開場那條 InfoNCE,只是負樣本改成從 queue 撈——跟前一個比,差別就在這條 queue 加動量編碼器。記憶點:小 batch 也能有海量又一致的負樣本。",
    "講到 queue 時指圖中佇列方塊。「可是等等」這裡刻意停頓、皺眉,製造回扣懸念,再揭曉動量編碼器這招。"),
  moco_v2: N(
    "MoCo v2 這頁很快。它沒有什麼新的大理論,loss 跟架構都跟 v1 一模一樣,就是一次很務實的工程升級:把對手 SimCLR 那邊驗證有效的三樣好料——MLP 投影頭(在編碼器後面再接一個小網路、把特徵投影到算 loss 的空間,只在訓練時用、評估就丟掉)、更強的高斯模糊增強、還有 cosine 學習率排程——直接搬進 MoCo 的框架裡。結果就是,在小資源、小 batch 的條件下,逼近了要用 TPU 大 batch 的 SimCLR。記憶點:好的點子會互相偷,而且偷得理直氣壯。這也預告了我們接下來要看的對手。",
    "語氣輕快帶笑。「互相偷」這句故意俏皮,逗一下觀眾。手指向下一頁方向。"),
  simclr_v1: N(
    "好,換另一派出場,Hinton 團隊的 SimCLR,理念跟 MoCo 完全相反。他們說:我幹嘛搞什麼佇列、什麼動量?太麻煩了。我就把 batch 開到超級大,大到光是這一個 batch 裡面,就有幾千個負樣本可以互相比,根本不需要存歷史。架構簡單到不行:就是兩條共享權重的分支、各接一個 MLP 投影頭,loss 叫 NT-Xent——名字嚇人,其實就是開場那條 InfoNCE 把特徵先做 L2 正規化後的版本。但他們發現一個關鍵——強增強才是真正的靈魂,尤其是夠狠的 color jitter 加高斯模糊,沒有它整個就學不起來。記憶點,也是它的痛點:這條路要 work,你得有 TPU 等級的超大 batch,一般實驗室根本玩不起。所以你看,MoCo 跟 SimCLR 這兩派之爭,本質上就是同一個問題——負樣本從哪來——的兩種哲學:一邊省記憶體用 queue,一邊用蠻力堆 batch。",
    "出場時語氣轉強、像介紹對手登場。講「太麻煩了」時模仿不屑表情。結尾把兩派並排對比,左右手各代表一派,做總結手勢。"),
  simclr_v2: N(
    "SimCLR v2 是同一條路的加強版,loss 完全沒變、還是 NT-Xent,主軸只在「規模」。兩個改動:投影頭從兩層加深到三層,backbone 換成更大更深的模型。它真正想證明的事情很有意思——大型自監督模型,其實是非常強的半監督學習者:先用海量無標籤資料把大模型 pretrain 好,再用極少量的標籤微調、最後「蒸餾」到小模型(蒸餾就是讓訓練好的大模型當老師、去教一個小模型,把能力壓縮進去)。記憶點:模型越大,從無標籤資料裡榨出來的東西越多。",
    "用手比「加深、變大」的動作。講到「只要一點點標籤」時用拇指食指比一個很小的縫隙。"),
  swav: N(
    "到這裡,有人開始覺得「兩兩比較」這件事本身就很笨重。SwAV 跳出來說:我不要再一對一去比哪張圖跟哪張圖了。架構上它換了打法:設一組 prototypes,可以想成一群可學習的群心;每張圖去算它比較接近哪些群,得到一個「分配」(也叫 code)。loss 也跟前面不一樣——不再是 InfoNCE,而是 swapped prediction:拿一個視角算出的特徵,去預測「另一個視角」的群分配,兩邊互換著猜,用的是交叉熵。訣竅在於它用 Sinkhorn 演算法,強迫這些分配要均勻攤平到每個群,不准全部擠到同一個群——這一步本身就在防坍塌。再加上 multi-crop,一張圖切兩大塊加好幾小塊,等於免費多了好多視角。記憶點最關鍵:它完全不需要成對的負樣本了。等一下,不用負樣本也能不坍塌?這句話,正好把我們推向最後的懸念。",
    "講 prototypes 時手在空中畫幾個群心。「不用負樣本也能不坍塌?」這句放慢、上揚,故意當作鉤子。"),
  infomin: N(
    "在收這一幕之前,InfoMin 提了一個很哲學、但超級重要的問題。先說它跟前面的差異:它沿用 SimCLR 的 backbone 跟 NT-Xent,唯一動手腳的是「餵進去的視角」。前面大家都在卷負樣本、卷架構,它卻退一步問:我們一直在拉近兩個增強視角,那到底——什麼樣的視角才是「好」的視角?它的答案叫 minimal sufficient,最小充分:兩個視角之間,該共享的語意要全部留住,但不該共享的、像顏色、紋理這種捷徑,要盡量砍掉。怎麼砍?用更激進的資料增強去破壞那些捷徑,逼模型不能偷懶。記憶點:有時候瓶頸不在 loss、不在架構,而在你餵進去的「視角」本身。好,第二幕到這裡。我們看到負樣本被玩出各種花樣,可是 SwAV 已經偷偷暗示了——負樣本根本不是唯一解。那如果,我們把這個念頭推到極致,乾脆連一個負樣本都不要呢?下一幕見。",
    "「退一步問」時身體微微後仰,做出思考姿態。最後三句語速放慢、加重,「連一個負樣本都不要呢」拋給觀眾後停頓兩秒,再切轉場。"),
  act3: N(
    "好,我們走到整個故事最瘋狂的一幕了。前面我們一路在拿拐杖:先拿掉人工標籤、再拿掉巨大的 batch、拿掉那個 memory queue。但有一根拐杖,從第一天到現在,誰都不敢碰——就是「負樣本」。負樣本是幹嘛的?它是那個「推開」的力量。對比學習的精神就是一句話:拉近自己、推開別人。那如果我把「推開別人」整個拿掉呢?你想想看,如果只剩下「拉近」、沒有任何「推開」,模型最聰明的偷懶方式是什麼?就是把所有東西都對應到同一個點——所有圖片都長一樣,loss 直接歸零,完美收工。這就是我們從頭到尾最怕的那個惡夢:坍塌,collapse。所以當年大家的共識是:沒有負樣本,一定坍塌。沒有例外。記住這個共識——因為接下來這幾篇,就是來打臉它的。",
    "走到舞台中央,停頓一下製造儀式感。講到「沒有負樣本一定坍塌」時放慢、加重,停 1 秒再翻頁。手可以做一個「推開」再「縮成一點」的動作。"),
  byol: N(
    "第一個跳出來的是 BYOL。它直接說:我不要負樣本,一個都不要。全場都在等著看它坍塌。它怎麼做的?架構上有兩條分支,一條叫 online、一條叫 target。重點來了:online 這邊特別多接一個小網路,叫 predictor,刻意讓兩條分支不對稱;而 target 不自己學,它是 online 的「慢動作影分身」,用動量(EMA、把過去權重慢慢平均)跟上,而且 target 這端切斷梯度。loss 也從對比式整個換掉——不再推開誰,而是讓 online 經過 predictor 去「預測」target 的特徵、把兩者拉近(用 MSE 均方誤差)。跟前面所有方法最大的差別就是:它一個負樣本都沒有。結果呢?它就是沒坍塌,而且效果好到嚇人。當時大家第一反應是「這一定哪裡有 bug」。記憶點就一句:靠不對稱,而不是靠推開,也能撐住不坍塌。負樣本這根拐杖,第一次被丟掉了。",
    "翻到 BYOL 頁,指架構圖上的 predictor 小框。講「慢動作影分身」時可比手勢。「沒坍塌」三個字加重,露出一點驚訝表情,把觀眾的好奇勾起來。"),
  simsiam: N(
    "BYOL 之後大家鬆一口氣,想說好,原來是那個動量 EMA 在偷偷防坍塌。結果何愷明團隊出來說:不,你們連這個都想多了。SimSiam 把動量 EMA 也砍掉,兩條分支直接共享同一個網路,沒有影分身、沒有 queue,什麼都沒有——架構比 BYOL 更精簡,差別就在它連 EMA target 都拿掉了,只留 predictor 跟那一刀。那它靠什麼不坍塌?就靠一個動作——stop-gradient,梯度在 target 那一端停下來、不往回傳。loss 則是負餘弦相似度(就是讓兩邊特徵的方向盡量對齊、愈像愈好)。就這樣。它等於是做了一個最小化的實驗,把所有東西都拿光,只留一根,然後指著它說:看,真正防坍塌的關鍵,從頭到尾就是這個 stop-gradient。乾淨、漂亮,一刀見血。",
    "翻頁。講「全部砍掉」時手做連續刪除動作。講到 stop-gradient 時指圖上那個 ⊘ 停止符號,停半秒,強調「就這一根」。"),
  barlow_twins: N(
    "Barlow Twins 更有意思,它不跟你玩拉近推開了,它換了一整套哲學。架構還是兩條共享權重的分支,只是後面接一個維度開得很寬的投影頭(到八千多維,這對它特別重要)。它說:我來看兩個視角嵌入之間的「互相關矩陣」——白話講,就是把「兩邊每一個特徵維度彼此有多相關」量出來、排成一張方表。你只要逼這個矩陣變成單位矩陣就好——對角線是 1,代表同一張圖的兩個視角要一致;非對角線壓到 0,代表每個維度不要互相重複、不要講同一件事。這叫「消除冗餘」。神奇的是,它沒有負樣本、沒有 EMA、也沒有 predictor,全部的功夫都在 loss 上。所以跟剛剛的 SimSiam 比,它連 stop-gradient 那一刀都不用——SimSiam 是靠停梯度打破對稱,Barlow Twins 則是直接在 loss 裡逼互相關矩陣去冗餘,完全是另一條路。那到這裡,我們把三篇放在一起看,會看出一條共同的線索:防坍塌的關鍵,從來不是「推開別人」,而是「打破對稱」。不管你用 predictor、用 stop-gradient、還是逼互相關矩陣去冗餘,本質上都是同一件事——不讓兩邊塌成一模一樣。這就是 ACT 3 留給我們的那句話。",
    "翻頁,指 loss 公式 C→I。講「打破對稱」這個收束句時放慢、看向全場,這是這一幕的結論,要讓它沉下去。"),
  act4: N(
    "好,到這裡我們的拐杖幾乎全拿光了:標籤、大 batch、queue、負樣本,一根一根丟掉。但有一件事一直沒變——底下那台引擎,一直都是 CNN、是 ResNet。那同一個時間,影像界正在發生一場大地震:Transformer 打進來了,ViT 出現了。問題就來了:如果我把對比學習底下的引擎,整個換成 Transformer,會發生什麼事?是無痛接上,還是又會炸出新的坍塌?我們最後一幕,就是看這群人怎麼把這台新引擎馴服,然後一路衝到今天的基礎模型。",
    "走回舞台中央。語氣轉成「最後一幕」的收尾感。講「換成 Transformer」時停一下,丟出懸念再翻頁。"),
  moco_v3: N(
    "第一個吃螃蟹的是 MoCo v3。它把對比學習直接搬到 ViT 上——架構大致還是 MoCo 那套 query/key 兩個編碼器,差別就在 backbone 從 ResNet 換成了 Transformer。結果一訓練就發現:超不穩,loss 會突然亂跳,訓到一半整個爛掉。團隊抓了很久,最後抓到一個超反直覺的兇手——是最前面那層 patch embedding,就是把圖片切成小塊、投影成向量的那一層。他們的解法簡單到誇張:把那一層直接凍結,不訓練它。一凍,整個就穩了。順手他們也把那個 memory queue 丟掉,loss 改回 SimCLR 那種 batch 內互相當負樣本的對稱 InfoNCE。記憶點:有時候穩定性的關鍵,不在你拚命調的地方,而在最不起眼的第一層。",
    "翻頁,指架構圖最底下的 patch embedding。講「凍結那一層」時做一個「定住」的手勢。「最不起眼的第一層」可以挑眉、語帶玄機。"),
  dino: N(
    "再來是 DINO,我個人最愛的一篇。它跟 MoCo v3 一樣都在 ViT 上,但更進一步——連負樣本都不要了。它玩的是「自我蒸餾」:一個 student、一個 teacher,但 teacher 不是另外請的高手,它就是 student 的影分身(一樣用 EMA 慢慢平均得到)。student 努力去預測 teacher 的輸出,loss 是一個交叉熵——讓 student 的輸出機率去逼近 teacher 的那個分布。為了不坍塌,它在 teacher 那邊加了兩個小動作:centering,把輸出拉回中心、不讓某一類獨大;sharpening,把分布變尖銳、逼它做決定。一個拉一個推,剛好卡在中間不塌。重點是,它完全沒有負樣本。但 DINO 最炸的不是分數——是這個。你看它的注意力圖(attention map,模型內部「在看哪裡」的熱力圖):沒有人教它什麼是物體、沒給任何標註,它的注意力居然自己聚焦在整隻狗、整隻鳥的身上、把整塊物體圈出來。它無意間學會了「分割物體」。這就是自監督最迷人的「啊哈」時刻。",
    "翻頁。講完機制後停頓,再切到 attention map 那張圖,指圖上亮起來的物體輪廓。「自己亮起來」放慢、語氣帶驚喜,讓觀眾跟著哇一下。"),
  dinov2: N(
    "最後一站,DINOv2。它其實沒有發明全新的招式——架構跟 loss 基本上就是把 DINO 的自蒸餾,再疊上一個叫 iBOT 的 loss:iBOT 把一部分 patch 遮起來、要模型去預測那些被遮位置的 teacher 特徵(等於影像版的「克漏字」,只是要補回來的是特徵、不是像素)。它真正做的是另一件更難的事——把對的方法放大:餵進一個精心篩選、上億張的資料集 LVD-142M,大力出奇蹟。結果就是一個真正通用的「視覺基礎模型」:你拿它的特徵,不用再訓練,直接接到分類、分割、深度估計,各種任務都打得很好。這就把我們整條線,接上了今天大家天天在講的 foundation model 浪潮。所以回頭看這五年——我們從「需要人標每一張圖」,一路走到「不需要標籤、不需要負樣本,只要餵夠多圖,模型自己就長出對世界的理解」。拐杖,一根一根,全丟掉了。",
    "翻頁,這是本段最後一張。講最後「拐杖全丟掉了」時放慢、看向全場做收尾,雙手做一個「攤開、放下」的動作,把這一段落乾淨地交回給主線。"),
  dinov3: N(
    "再補一個 2025 的續章:DINOv3,一樣是 Meta。它跟 DINOv2 是同一條路——架構、loss 幾乎一模一樣,都是 DINO + iBOT 自蒸餾——差別在兩件事。第一,規模又往上爆:teacher 拉到 70 億參數、餵 17 億張影像。第二,加了一招 Gram anchoring,專門解一個老問題:模型訓練很久之後,patch 級的 dense 特徵會慢慢退化、變糊,Gram anchoring 就是去把它穩住。結果就是它在分割、深度、找對應點這種 dense 任務上又上一層樓——凍結 backbone、完全不微調,單一模型就打贏一票專門訓練的模型。右邊這張 UMAP,就是它官方權重在 CIFAR-10 上抽出來的特徵,你看分得多乾淨。所以 DINO 這條自蒸餾線,從 v1 到 v3,其實就是同一個點子不斷放大、把通用視覺基礎模型一路推到新高度。",
    "切到 DINOv3 頁。一句話定位「DINOv2 的放大版 + Gram anchoring」。指右邊官方權重 UMAP,強調分群乾淨。這是補充/續章頁,節奏明快;若時間不夠,可整頁略過、只留一句『DINOv3 又把規模推得更大』。"),
  collapseTable: N(
    "好,我們一路從 2018 講到現在,看了十四個方法、四個時代,名字一個比一個酷。但我現在要把整桌菜端走,只留一句話。你們還記得我開場埋的那個伏筆嗎?我說過,整個對比學習最怕的,就是「坍塌」——模型偷懶,把所有圖片都對應到同一個點,loss 漂亮歸零,但表徵一文不值。各位看這張表。我把每一代的代表方法、它用不用負樣本、還有它「靠什麼防坍塌」全列出來。你會發現一件很驚人的事:Era 1 用 memory bank 的大量負樣本把表徵推開;Era 2 的 MoCo、SimCLR 換成 queue 跟大 batch 的負樣本;然後 SwAV 開始叛逆,改用 prototype 加 Sinkhorn 把點均勻分散;到 Era 3,BYOL 跟 SimSiam 乾脆不要負樣本了,靠 EMA、靠 stop-gradient 打破對稱;Barlow Twins 用去相關;DINO 用 centering 加 sharpening。機制全都不一樣,對吧?但是——它們其實都在回答同一個問題。這就是今天的 punchline:十四個方法,本質上都在回答這一句——「沒有標籤,我到底要怎麼樣,才能不讓表徵坍塌?」五年的演化史,說穿了,就是人類想出十四種不同的辦法,去回答這同一個問題。",
    "停頓一拍再開講,回扣開場。講到「坍塌」時用手做一個「全部擠成一團」的手勢。逐行指表格最後一欄(防坍塌機制),讓大家看到機制各不同。講最後一句 punchline 時放慢、加重,並用雷射筆圈住底部那行「沒有標籤,怎麼不讓表徵坍塌?」"),
  prog1: N(
    "講了這麼多理論,你可能會想:好啦,講得很玄,那「防坍塌成功」到底長什麼樣子?這幾張,是我自己跑出來的。我在學校的兩張 H100 上,用 CIFAR-10、ResNet-18,從零開始硬訓兩百個 epoch,然後在不同 epoch 各拍一張特徵的 UMAP 快照,串成這條演進帶。先看 Era 1 到 Era 2 這四個:Instance Discrimination、Invariant Spread、SimCLR v1、SimCLR v2。各位看最左邊——epoch 0,隨機初始化,就是一坨,十個類別全糊在一起,這其實就是「坍塌的起點」。然後往右看,隨著訓練,同一類的點慢慢聚過來,不同類的慢慢被推開,到最右邊 epoch 200,已經能看到清楚的群落。注意 SimCLR 這兩條,分群的速度明顯比較俐落、邊界比較乾淨——這就是 in-batch 大量正負樣本對比的威力。沒有用到任何一個標籤,純靠「同張圖的兩個視角要像、跟別張圖要不像」這個訊號,硬是把結構學出來了。",
    "切到第一張演進帶。用雷射筆從最左 epoch 0 那坨點,由左往右掃到 epoch 200 的分群,配合「一坨 → 散開 → 聚成群」的手勢。講 SimCLR 比較俐落時,特別指那兩排。語氣帶一點「這是我親手跑的」的驕傲。"),
  prog2: N(
    "再來看 Era 2 到 Era 3 這三個,重點來了:InfoMin、BYOL、還有 SimSiam。前面那些都還有負樣本當靠山,但 BYOL 跟 SimSiam——記得嗎?它們是完全不用負樣本的那兩個。理論上最容易坍塌的,就是它們。可是你看這兩排演進帶,從 epoch 0 那坨,一樣穩穩地分化成漂亮的群落,完全沒有塌成一個點。這就是眼見為憑:原來光靠 stop-gradient、光靠一個 predictor 加 EMA,真的就能防坍塌。它的分群型態跟 SimCLR 那種對比式的稍微不一樣,但「能分群」這件事,這張圖直接幫它們作證了。",
    "切到第二張演進帶。先指 BYOL、SimSiam 兩排,語氣上揚製造「它們居然沒塌」的驚喜感。可以回頭半秒指一下前一張,做對照。節奏要快,這頁是過場。"),
  prog3: N(
    "前面那七條,是我第一批從零訓的。後來我把原本用官方權重展示的六個方法——MoCo v1、v2、SwAV、Barlow Twins,還有搬上 ViT 的 MoCo v3 跟 DINO——也全部從零自己訓了一遍,把整套湊齊。這一頁就是它們的逐-epoch 演進。一樣從 epoch 0 的一坨,全部穩穩分化成清楚的群落;特別是最後那兩排 MoCo v3 跟 DINO,它們底下是 Transformer、之前最怕訓練發散,結果一樣乾乾淨淨分群。到這裡,十三個自監督方法,全部都用親手跑的動畫幫自己作證了。",
    "切到第三張演進帶。一句話帶出「這六個是後來補訓、湊齊整套」。指最後 MoCo v3、DINO 兩排,強調「連 Transformer 都穩穩分群」。節奏快,這是補完性質的過場頁。"),
  liveDemo: N(
    "好,光看投影片不夠過癮,我們來看現場的。這就是整個 demo 最核心的一張對照圖:左邊是訓練前、隨機初始化的特徵,你看,就是一團毛球,紅紅綠綠全部混在一起,根本分不出來誰是誰——這就是「坍塌」的長相。右邊呢,是充分訓練之後的特徵空間,十個顏色,乾乾淨淨各自成群。從「一團」變「分群」,這就是自監督學習在做的事情,全程沒有用到任何一個標籤。這裡我要特別澄清一個大家很容易誤會的點,這也是我跑完所有實驗最大的體會:「分群分得好不好」,主要看的是——你有沒有「充分訓練」。只要 epoch 夠、訓練到位,幾乎每個方法最後都能把群分出來。那方法之間的差異在哪?差在「架構」跟「loss 怎麼設計」——也就是它用哪一種招式來防坍塌、收斂得多快、需要多大的 batch、要不要負樣本。換句話說,方法的不同,不是「能不能分群」的差別,而是「用什麼代價、走哪條路去防坍塌」的差別。我這邊放了 GIF,我直接讓它跑一次,大家看著那團點,慢慢、慢慢地,散開、歸隊。",
    "切到 LIVE DEMO 頁。先指左邊紅框「訓練前」說「一團毛球」,再指右邊綠框「訓練後」說「分群」,手沿著中間的箭頭由左滑到右。講「充分訓練」時加重、停頓。然後實際播放 GIF/UMAP 動畫,邊播邊用手跟著點群移動,留幾秒讓大家看動畫,不要一直講話蓋過去。"),
  closing: N(
    "好,我們收尾。如果今天這場報告,你只能帶走一句話,我希望是這句:這五年來,對比學習一直在做同一件事——「拿掉依賴」。從一開始要 memory bank 這根拐杖,到後來連負樣本都不要了,最後連 CNN 這個架構本身都被換成 Transformer。一步一步,把所有外部的拐杖通通拿掉。給大家三個帶得走的重點:第一,所有方法其實共用同一個 InfoNCE 的直覺——拉近正樣本、推開負樣本,名字再花俏,骨子裡都是這個。第二,這也是我最想強調的:真正的難題從頭到尾都是「防坍塌」,負樣本只是解法之一,不是唯一的路。第三,整個趨勢就是:用越來越少的假設、越來越大的規模,一路走向今天的 foundation model。最後打個廣告——今天講的這十四個方法,我全部都實作成一個開源教學專案,統一的程式架構,每一個 loss 都能單獨拿出來讀、自己跑訓練,剛剛那些 H100 的結果你都可以重現。歡迎大家 clone 回去玩。謝謝大家的聆聽,接下來歡迎提問!",
    "切到結尾頁。講「拿掉依賴」那句時放最慢、最重,這是全場的題眼。三個 takeaway 配合螢幕上的 1、2、3 圓點,逐點手指過去。講開源專案時可指底部那行字(或亮出 GitHub QR/連結)。最後一句「歡迎提問」面向全場、微笑、稍微張開手,正式進入 Q&A。"),
};

// ---------- reusable bits ----------
function timelineRibbon(slide, y) {
  // a thin ribbon with 4 era nodes + years, used as a footer motif
  const eras = [
    { yr: "2018", c: E1, lab: "proxy" },
    { yr: "2020", c: E2, lab: "contrastive" },
    { yr: "2021", c: E3, lab: "no-negative" },
    { yr: "2021+", c: E4, lab: "transformer" },
  ];
  const x0 = 0.9, x1 = PW - 0.9;
  slide.addShape(pres.shapes.LINE, { x: x0, y: y, w: x1 - x0, h: 0, line: { color: "33486E", width: 1.5 } });
  const step = (x1 - x0) / 3;
  eras.forEach((e, i) => {
    const cx = x0 + step * i;
    slide.addShape(pres.shapes.OVAL, { x: cx - 0.09, y: y - 0.09, w: 0.18, h: 0.18, fill: { color: e.c }, line: { color: NAVY, width: 1.5 } });
    slide.addText(e.yr, { x: cx - 0.6, y: y + 0.12, w: 1.2, h: 0.26, align: "center", fontFace: FACE, fontSize: 11, bold: true, color: e.c, margin: 0 });
    slide.addText(e.lab, { x: cx - 0.7, y: y + 0.38, w: 1.4, h: 0.22, align: "center", fontFace: MONO, fontSize: 8.5, color: ICE, margin: 0 });
  });
}

function eraChip(slide, x, y, label, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.95, h: 0.42, rectRadius: 0.21, fill: { color }, shadow: softShadow() });
  slide.addText(label, { x, y, w: 1.95, h: 0.42, align: "center", valign: "middle", fontFace: FACE, fontSize: 12.5, bold: true, color: WHITE, charSpacing: 1, margin: 0 });
}

function contentHeader(slide, chipLabel, color, title) {
  slide.background = { color: LIGHT };
  eraChip(slide, 0.7, 0.55, chipLabel, color);
  slide.addText(title, { x: 0.68, y: 1.05, w: 12, h: 0.85, fontFace: FACE, fontSize: 33, bold: true, color: INK, margin: 0 });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.72, y: 1.92, w: 0.55, h: 0.07, fill: { color } });
}

function actDivider(actNo, color, years, question, notes) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // big tinted act number block on the left
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: PH, fill: { color } });
  s.addText("ACT", { x: 0.9, y: 2.0, w: 3, h: 0.5, fontFace: MONO, fontSize: 20, bold: true, color: color, charSpacing: 6, margin: 0 });
  s.addText(actNo, { x: 0.78, y: 2.35, w: 4.2, h: 2.6, fontFace: FACE, fontSize: 150, bold: true, color: WHITE, margin: 0 });
  s.addText(years, { x: 5.0, y: 2.55, w: 7.5, h: 0.5, fontFace: MONO, fontSize: 17, bold: true, color: color, margin: 0 });
  s.addText(question, { x: 5.0, y: 3.05, w: 7.6, h: 1.9, fontFace: FACE, fontSize: 31, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.1 });
  timelineRibbon(s, 6.7);
  if (notes) s.addNotes(notes);
  return s;
}

// ---------- per-paper slide template + architecture diagram ----------
function abox(s, x, y, w, h, text, fill, tcol, bcol) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: bcol || fill, width: 1 }, shadow: softShadow() });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: FACE, fontSize: 9.5, bold: true, color: tcol, margin: 0, lineSpacingMultiple: 0.95 });
}
function hArrow(s, x, y, w) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: "90A0C0", width: 1.5, endArrowType: "triangle" } });
}
function diagArrow(s, x, yTop, w, h, up) {
  s.addShape(pres.shapes.LINE, { x, y: yTop, w, h, flipV: !!up, line: { color: "90A0C0", width: 1.25, endArrowType: "triangle" } });
}
const BOXF = "EAF0FA", BOXB = "C3D2EA";

// generic two-branch (siamese) architecture; cfg flags morph it per method
function siamese(s, o, cfg) {
  const rowH = 0.5;
  const midY = o.y + o.h / 2;
  const topY = o.y + 0.04, botY = o.y + o.h - rowH - 0.04;
  const topMid = topY + rowH / 2, botMid = botY + rowH / 2;
  const ix = o.x, iw = 0.62;
  abox(s, ix, midY - 0.3, iw, 0.6, "輸入\nx", NAVY3, WHITE);
  const ex = o.x + 1.45, ew = 1.3;

  if (cfg.single) {
    hArrow(s, ix + iw, midY, ex - (ix + iw));
    abox(s, ex, midY - rowH / 2, ew, rowH, cfg.top, BOXF, INK, BOXB);
    let cx = ex + ew;
    if (cfg.proj) { hArrow(s, cx, midY, 0.5); abox(s, cx + 0.5, midY - rowH / 2, 0.85, rowH, "投影\ng", BOXF, INK, BOXB); cx += 0.5 + 0.85; }
    const lx = cx + 0.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: midY - 0.32, w: 1.55, h: 0.64, rectRadius: 0.08, fill: { color: NAVY }, shadow: makeShadow() });
    s.addText(cfg.loss, { x: lx, y: midY - 0.32, w: 1.55, h: 0.64, align: "center", valign: "middle", fontFace: FACE, fontSize: 10.5, bold: true, color: ACCENT, margin: 0 });
    hArrow(s, cx, midY, lx - cx);
    if (cfg.extra) {
      const exx = lx + 1.55 + 0.55;
      abox(s, exx, midY - 0.32, 1.7, 0.64, cfg.extra, E2, NAVY);
      s.addShape(pres.shapes.LINE, { x: lx + 1.55, y: midY, w: exx - (lx + 1.55), h: 0, line: { color: "90A0C0", width: 1.5, beginArrowType: "triangle" } });
    }
    return;
  }

  diagArrow(s, ix + iw, topMid, ex - (ix + iw), midY - topMid, true);
  diagArrow(s, ix + iw, midY, ex - (ix + iw), botMid - midY, false);
  abox(s, ex, topY, ew, rowH, cfg.top, BOXF, INK, BOXB);
  abox(s, ex, botY, ew, rowH, cfg.bot, BOXF, INK, BOXB);
  if (cfg.botNote) s.addText(cfg.botNote, { x: ex, y: botY + rowH + 0.01, w: ew, h: 0.22, align: "center", fontFace: MONO, fontSize: 8, bold: true, color: cfg.noteColor || MUTE, margin: 0 });
  let cx = ex + ew;
  if (cfg.proj) {
    hArrow(s, cx, topMid, 0.45); hArrow(s, cx, botMid, 0.45);
    abox(s, cx + 0.45, topY, 0.8, rowH, "投影\ng", BOXF, INK, BOXB);
    abox(s, cx + 0.45, botY, 0.8, rowH, "投影\ng", BOXF, INK, BOXB);
    cx += 0.45 + 0.8;
  }
  let zxTop = cx;
  if (cfg.pred) {
    hArrow(s, cx, topMid, 0.45);
    abox(s, cx + 0.45, topY, 0.95, rowH, "predictor\nq", E3, WHITE);
    zxTop = cx + 0.45 + 0.95;
  }
  const gap0 = (cfg.sgBot && !cfg.pred) ? 1.45 : 0.55;
  const lx = Math.max(zxTop, cx) + gap0;
  const lossTop = topMid - 0.06, lossH = (botMid - topMid) + 0.12;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: lossTop, w: 1.55, h: lossH, rectRadius: 0.08, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText(cfg.loss, { x: lx, y: lossTop, w: 1.55, h: lossH, align: "center", valign: "middle", fontFace: FACE, fontSize: 10.5, bold: true, color: ACCENT, margin: 0, lineSpacingMultiple: 1.0 });
  hArrow(s, zxTop, topMid, lx - zxTop);
  hArrow(s, cx, botMid, lx - cx);
  if (cfg.sgBot) s.addText("⊘ stop-grad", { x: cx + 0.1, y: botMid - 0.34, w: (lx - cx) - 0.2, h: 0.24, align: "center", fontFace: MONO, fontSize: 8.5, bold: true, color: E3, margin: 0 });
  if (cfg.extra) {
    const exx = lx + 1.55 + 0.5;
    abox(s, exx, midY - 0.32, 1.75, 0.64, cfg.extra, E2, NAVY);
    s.addShape(pres.shapes.LINE, { x: lx + 1.55, y: midY, w: exx - (lx + 1.55), h: 0, line: { color: "90A0C0", width: 1.5, beginArrowType: "triangle" } });
  }
}

function methodSlide(spec) {
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  eraChip(s, 0.7, 0.5, spec.era.chip, spec.era.color);
  s.addText(spec.name, { x: 0.68, y: 0.97, w: 9.0, h: 0.66, fontFace: FACE, fontSize: 29, bold: true, color: INK, margin: 0 });
  s.addText(spec.venue, { x: 9.5, y: 1.05, w: 3.1, h: 0.5, align: "right", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: spec.era.color, margin: 0 });
  s.addText("📄 " + spec.authors, { x: 0.72, y: 1.62, w: 11.85, h: 0.3, fontFace: FACE, fontSize: 10.5, italic: true, color: MUTE, margin: 0 });
  const onEra = (spec.era.color === E1 || spec.era.color === E2) ? NAVY : WHITE;

  // diagram card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.04, w: 11.93, h: 2.02, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  s.addText("架構示意", { x: 0.88, y: 2.12, w: 3, h: 0.26, fontFace: FACE, fontSize: 10, bold: true, color: MUTE, charSpacing: 1, margin: 0 });
  spec.diagram(s, { x: 1.05, y: 2.46, w: 11.3, h: 1.5 });

  // --- bottom: three columns (text · loss/contribution · UMAP demo) ---
  // col1: text card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.22, w: 5.35, h: 2.73, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  s.addText("核心點子", { x: 0.92, y: 4.34, w: 4.95, h: 0.28, fontFace: FACE, fontSize: 12, bold: true, color: spec.era.color, margin: 0 });
  s.addText(spec.idea, { x: 0.92, y: 4.62, w: 4.95, h: 0.74, fontFace: FACE, fontSize: 11.5, color: INK, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText("關鍵機制", { x: 0.92, y: 5.42, w: 4.95, h: 0.28, fontFace: FACE, fontSize: 12, bold: true, color: spec.era.color, margin: 0 });
  s.addText(spec.mechanism.map((m, i) => ({ text: m, options: { bullet: { code: "2022" }, breakLine: i < spec.mechanism.length - 1, color: MUTE } })),
    { x: 0.92, y: 5.72, w: 4.95, h: 1.15, fontFace: FACE, fontSize: 10.5, paraSpaceAfter: 5, lineSpacingMultiple: 1.0, margin: 0 });

  // col2: loss panel + contribution
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 4.22, w: 3.05, h: 1.5, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText("LOSS", { x: 6.4, y: 4.31, w: 2.6, h: 0.24, fontFace: MONO, fontSize: 9, bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  s.addText(spec.loss, { x: 6.26, y: 4.55, w: 2.94, h: 0.74, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText(spec.lossNote, { x: 6.28, y: 5.33, w: 2.9, h: 0.32, align: "center", fontFace: FACE, fontSize: 9, italic: true, color: ICE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 5.88, w: 3.05, h: 1.07, fill: { color: spec.era.color } });
  s.addText("貢獻 / 關鍵", { x: 6.4, y: 5.95, w: 2.7, h: 0.26, fontFace: FACE, fontSize: 10, bold: true, color: onEra, charSpacing: 1, margin: 0 });
  s.addText(spec.contribution, { x: 6.4, y: 6.22, w: 2.75, h: 0.68, fontFace: FACE, fontSize: 11, bold: true, color: onEra, margin: 0, lineSpacingMultiple: 1.03 });

  // col3: per-method UMAP demo
  const hasImg = spec.demo && fs.existsSync(spec.demo);
  const TRAINED = ["instance_discrimination", "invariant_spread", "simclr_v1", "simclr_v2", "infomin", "byol", "simsiam", "moco_v1", "moco_v2", "swav", "barlow_twins", "moco_v3", "dino"];
  const dkey = spec.demo ? spec.demo.split("/").pop().replace(/\.(png|gif)$/, "") : "";
  const isTrained = TRAINED.includes(dkey);
  const isGif = /\.gif$/.test(spec.demo || "");
  s.addShape(pres.shapes.RECTANGLE, { x: 9.4, y: 4.22, w: 3.23, h: 2.73, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText(!hasImg ? "LIVE DEMO · 待訓練" : (isTrained ? "LIVE DEMO · 自訓動畫 UMAP" : "LIVE DEMO · 官方權重 UMAP"), { x: 9.5, y: 4.31, w: 3.05, h: 0.26, align: "center", fontFace: MONO, fontSize: 8, bold: true, color: ACCENT, charSpacing: 1, margin: 0 });
  if (hasImg) {
    s.addImage({ path: spec.demo, x: 9.965, y: 4.62, w: 2.1, h: 2.1, sizing: { type: "contain", w: 2.1, h: 2.1 } });
    s.addText(isTrained ? (isGif ? "CIFAR-10 · 自訓 200ep · GIF（PPT 365 播放）" : "CIFAR-10 · 自訓 200 epoch（H100）") : "官方釋出之預訓練權重 · CIFAR-10 特徵", { x: 9.45, y: 6.66, w: 3.15, h: 0.24, align: "center", fontFace: FACE, fontSize: 7.5, italic: true, color: MUTE, margin: 0 });
  } else {
    s.addText("⏳", { x: 9.4, y: 5.02, w: 3.23, h: 0.55, align: "center", fontFace: FACE, fontSize: 28, color: ICE, margin: 0 });
    s.addText("待 GPU 充分訓練後\n補上該方法的 UMAP", { x: 9.5, y: 5.68, w: 3.05, h: 0.7, align: "center", valign: "top", fontFace: FACE, fontSize: 11, color: ICE, margin: 0, lineSpacingMultiple: 1.2 });
  }
  // speaker notes — keyed by the method's demo basename (instance_discrimination … dinov2)
  const _nk = spec.demo ? spec.demo.split("/").pop().replace(/\.(png|gif)$/, "") : "";
  if (NOTES[_nk]) s.addNotes(NOTES[_nk]);
}

// era presets
const A1 = (chip) => ({ chip, color: E1 });
const A2 = (chip) => ({ chip, color: E2 });
const A3 = (chip) => ({ chip, color: E3 });
const A4 = (chip) => ({ chip, color: E4 });

// ============================================================
// 1. COVER
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // faint corner accent
  s.addShape(pres.shapes.OVAL, { x: 10.4, y: -2.6, w: 5.5, h: 5.5, fill: { color: NAVY3, transparency: 35 } });
  s.addShape(pres.shapes.OVAL, { x: 11.6, y: 4.6, w: 4.2, h: 4.2, fill: { color: E2, transparency: 82 } });

  s.addText("自監督學習 · SELF-SUPERVISED LEARNING", {
    x: 0.95, y: 1.25, w: 11, h: 0.4, fontFace: MONO, fontSize: 14, bold: true, color: ACCENT, charSpacing: 3, margin: 0 });
  s.addText("對比學習的演化史", {
    x: 0.9, y: 1.75, w: 11.5, h: 1.3, fontFace: FACE, fontSize: 60, bold: true, color: WHITE, margin: 0 });
  s.addText("The Evolution of Contrastive Learning", {
    x: 0.95, y: 3.05, w: 11, h: 0.6, fontFace: FACE, fontSize: 24, color: ICE, margin: 0 });
  s.addText([
    { text: "從 2018 到今天 — 一場「", options: {} },
    { text: "不斷拿掉依賴", options: { color: ACCENT, bold: true } },
    { text: "」的旅程", options: {} },
  ], { x: 0.95, y: 3.75, w: 11, h: 0.5, fontFace: FACE, fontSize: 17, color: ICE, margin: 0 });

  timelineRibbon(s, 5.55);
  s.addText("Yi-Ting Li · 2026", { x: 0.95, y: 6.85, w: 6, h: 0.35, fontFace: MONO, fontSize: 11, color: MUTE, margin: 0 });
  s.addNotes(NOTES.cover);
}

// ============================================================
// 2. BIG PICTURE — one sentence
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("一句話看懂整個領域", { x: 0.9, y: 0.6, w: 11.5, h: 0.7, fontFace: FACE, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  s.addText([
    { text: "對比學習花了 5 年，一步步拿掉它對", options: {} },
    { text: "「外部拐杖」", options: { color: ACCENT, bold: true } },
    { text: "的依賴。", options: {} },
  ], { x: 0.9, y: 1.45, w: 11.6, h: 0.9, fontFace: FACE, fontSize: 23, color: ICE, margin: 0 });

  const steps = [
    { yr: "2018", t: "拿掉標籤", s: "proxy task", c: E1 },
    { yr: "2020", t: "拿掉 memory bank", s: "queue / 大 batch", c: E2 },
    { yr: "2021", t: "拿掉負樣本", s: "no-negative", c: E3 },
    { yr: "2021+", t: "換上 Transformer", s: "foundation model", c: E4 },
  ];
  const bx = 0.9, bw = 2.72, gap = 0.28, by = 3.05, bh = 2.5;
  steps.forEach((st, i) => {
    const x = bx + i * (bw + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: NAVY2 }, line: { color: st.c, width: 1.5 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: by, w: bw, h: 0.12, fill: { color: st.c } });
    s.addText(st.yr, { x: x + 0.25, y: by + 0.32, w: bw - 0.5, h: 0.5, fontFace: MONO, fontSize: 22, bold: true, color: st.c, margin: 0 });
    s.addText(st.t, { x: x + 0.25, y: by + 0.95, w: bw - 0.5, h: 0.85, fontFace: FACE, fontSize: 18.5, bold: true, color: WHITE, margin: 0, valign: "top" });
    s.addText(st.s, { x: x + 0.25, y: by + 1.95, w: bw - 0.5, h: 0.4, fontFace: MONO, fontSize: 11.5, color: ICE, margin: 0 });
    if (i < steps.length - 1) {
      s.addText("→", { x: x + bw - 0.06, y: by + 0.9, w: gap + 0.12, h: 0.6, align: "center", valign: "middle", fontFace: FACE, fontSize: 26, bold: true, color: ACCENT, margin: 0 });
    }
  });
  s.addText("每一步，都是前一代「撞到牆」之後翻牆的結果。", {
    x: 0.9, y: 6.05, w: 11.6, h: 0.5, fontFace: FACE, fontSize: 15, italic: true, color: ACCENT, margin: 0 });
  s.addNotes(NOTES.bigPicture);
}

// ============================================================
// 3. CORE INTUITION (light, two-column with diagram)
// ============================================================
{
  const s = pres.addSlide();
  contentHeader(s, "核心直覺", ACCENT.replace("22D3EE", "0E7490"), "對比學習在做什麼？");
  // left text
  s.addText([
    { text: "沒有標籤，從資料本身造出學習信號", options: { bullet: { code: "2022" }, breakLine: true, bold: true, color: INK } },
    { text: "同一張圖的兩個增強版本 = 正樣本對 → 拉近", options: { bullet: { code: "2022" }, breakLine: true, color: "2A6B4F" } },
    { text: "不同圖的特徵 = 負樣本 → 推開", options: { bullet: { code: "2022" }, breakLine: true, color: "B43050" } },
    { text: "學到「對增強保持不變」的語意特徵", options: { bullet: { code: "2022" }, color: INK } },
  ], { x: 0.7, y: 2.35, w: 6.0, h: 3.6, fontFace: FACE, fontSize: 17.5, color: INK, paraSpaceAfter: 16, lineSpacingMultiple: 1.05 });

  // right diagram panel
  const px = 7.15, py = 2.3, pw2 = 5.45, ph2 = 4.4;
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw2, h: ph2, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
  const cx = px + pw2 / 2, cy = py + ph2 / 2;
  // pull arrow + positive
  s.addShape(pres.shapes.LINE, { x: cx - 1.55, y: cy, w: 1.35, h: 0, line: { color: "2A8F5F", width: 2.5, endArrowType: "triangle", beginArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: px + 0.45, y: cy - 0.38, w: 0.76, h: 0.76, fill: { color: "2A8F5F" }, shadow: softShadow() });
  s.addText("正", { x: px + 0.45, y: cy - 0.38, w: 0.76, h: 0.76, align: "center", valign: "middle", fontFace: FACE, fontSize: 16, bold: true, color: WHITE, margin: 0 });
  s.addText("拉近", { x: cx - 1.7, y: cy - 0.62, w: 1.6, h: 0.3, align: "center", fontFace: FACE, fontSize: 12, bold: true, color: "2A8F5F", margin: 0 });
  // anchor
  s.addShape(pres.shapes.OVAL, { x: cx - 0.5, y: cy - 0.5, w: 1.0, h: 1.0, fill: { color: INK }, line: { color: ACCENT, width: 3 }, shadow: makeShadow() });
  s.addText("錨點", { x: cx - 0.5, y: cy - 0.5, w: 1.0, h: 1.0, align: "center", valign: "middle", fontFace: FACE, fontSize: 15, bold: true, color: WHITE, margin: 0 });
  // push arrows + negatives (two)
  s.addShape(pres.shapes.LINE, { x: cx + 0.62, y: cy - 0.85, w: 1.4, h: 0.7, flipV: true, line: { color: "C0395A", width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: cx + 0.62, y: cy + 0.15, w: 1.4, h: 0.7, line: { color: "C0395A", width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: cx + 2.0, y: cy - 1.15, w: 0.7, h: 0.7, fill: { color: "C0395A" }, shadow: softShadow() });
  s.addText("負", { x: cx + 2.0, y: cy - 1.15, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: FACE, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addShape(pres.shapes.OVAL, { x: cx + 2.0, y: cy + 0.45, w: 0.7, h: 0.7, fill: { color: "C0395A" }, shadow: softShadow() });
  s.addText("負", { x: cx + 2.0, y: cy + 0.45, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: FACE, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addText("推開", { x: cx + 0.72, y: cy - 0.15, w: 1.2, h: 0.3, align: "center", valign: "middle", fontFace: FACE, fontSize: 12, bold: true, color: "C0395A", margin: 0 });
  s.addNotes(NOTES.coreIntuition);
}

// ============================================================
// 4. InfoNCE — shared language
// ============================================================
{
  const s = pres.addSlide();
  contentHeader(s, "共同語言", "0E7490", "一條式貫穿所有方法：InfoNCE");
  // formula panel (dark) — rendered as a stacked fraction so nothing wraps
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.35, w: 11.9, h: 1.55, fill: { color: NAVY }, shadow: makeShadow() });
  s.addText("ℒ  =  − log", { x: 1.7, y: 2.35, w: 3.1, h: 1.55, align: "right", valign: "middle", fontFace: MONO, fontSize: 24, bold: true, color: WHITE, margin: 0 });
  s.addText("exp( sim(zᵢ, zⱼ) / τ )", { x: 5.1, y: 2.52, w: 6.6, h: 0.55, align: "center", valign: "bottom", fontFace: MONO, fontSize: 19, bold: true, color: "7CF0A0", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 5.35, y: 3.14, w: 6.1, h: 0, line: { color: WHITE, width: 1.5 } });
  s.addText("Σₖ  exp( sim(zᵢ, zₖ) / τ )", { x: 5.1, y: 3.2, w: 6.6, h: 0.55, align: "center", valign: "top", fontFace: MONO, fontSize: 19, bold: true, color: "FF8FA8", margin: 0 });

  const cards = [
    { t: "分子 = 正樣本", d: "同圖兩個增強的相似度\n→ 希望它「大」", c: "2A8F5F" },
    { t: "分母 = 負樣本", d: "與所有其他樣本的相似度\n→ 希望它「小」", c: "C0395A" },
    { t: "τ = 溫度", d: "控制分布尖銳程度\n→ 影響對難負樣本的關注", c: "0E7490" },
  ];
  const cw = 3.83, cg = 0.2, cy0 = 4.25, ch = 1.55, x0 = 0.7;
  cards.forEach((c, i) => {
    const x = x0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy0, w: cw, h: ch, fill: { color: CARD }, line: { color: "DCE3F0", width: 1 }, shadow: softShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy0, w: 0.1, h: ch, fill: { color: c.c } });
    s.addText(c.t, { x: x + 0.3, y: cy0 + 0.18, w: cw - 0.5, h: 0.4, fontFace: FACE, fontSize: 16, bold: true, color: c.c, margin: 0 });
    s.addText(c.d, { x: x + 0.3, y: cy0 + 0.6, w: cw - 0.5, h: 0.85, fontFace: FACE, fontSize: 12.5, color: MUTE, margin: 0, lineSpacingMultiple: 1.05 });
  });
  s.addText([
    { text: "接下來 14 種方法，全是這條式的 ", options: {} },
    { text: "「變形」", options: { bold: true, color: "0E7490" } },
    { text: " 或對它的 ", options: {} },
    { text: "「逃脫」", options: { bold: true, color: E3 } },
    { text: "。", options: {} },
  ], { x: 0.7, y: 6.1, w: 11.9, h: 0.5, align: "center", fontFace: FACE, fontSize: 15, italic: true, color: INK, margin: 0 });
  s.addNotes(NOTES.infonce);
}

// ============================================================
// 5. EVOLUTION MAP (dark) — the centerpiece
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("演化全景圖", { x: 0.9, y: 0.5, w: 11.5, h: 0.7, fontFace: FACE, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  s.addText("時間 →　每一代都在解決前一代的瓶頸", { x: 0.9, y: 1.18, w: 11.5, h: 0.4, fontFace: FACE, fontSize: 14, color: ICE, margin: 0 });

  const cols = [
    { yr: "2018–19", c: E1, head: "Era 1 · 代理任務", ms: ["Instance\nDiscrimination", "Invariant\nSpread", "memory bank\n+ NCE"] },
    { yr: "2020", c: E2, head: "Era 2 · 對比 / 分群", ms: ["MoCo v1 / v2", "SimCLR v1 / v2", "SwAV · InfoMin"] },
    { yr: "2020–21", c: E3, head: "Era 3 · 無負樣本", ms: ["BYOL", "SimSiam", "Barlow Twins"] },
    { yr: "2021+", c: E4, head: "Era 4 · Transformer", ms: ["MoCo v3", "DINO", "DINOv2"] },
  ];
  const cw = 2.78, cg = 0.32, x0 = 0.9, y0 = 2.0, ch = 4.3;
  cols.forEach((col, i) => {
    const x = x0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: cw, h: 0.6, fill: { color: col.c } });
    s.addText(col.yr, { x: x + 0.15, y: y0 + 0.08, w: cw - 0.3, h: 0.44, fontFace: MONO, fontSize: 15, bold: true, color: NAVY, margin: 0, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0 + 0.6, w: cw, h: ch - 0.6, fill: { color: NAVY2 }, line: { color: col.c, width: 1.5 } });
    s.addText(col.head, { x: x + 0.2, y: y0 + 0.75, w: cw - 0.4, h: 0.55, fontFace: FACE, fontSize: 14.5, bold: true, color: col.c, margin: 0 });
    let yy = y0 + 1.4;
    col.ms.forEach((m) => {
      const lines = m.split("\n").length;
      const mh = lines > 1 ? 0.78 : 0.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: yy, w: cw - 0.4, h: mh, rectRadius: 0.06, fill: { color: NAVY3 } });
      s.addText(m.replace("\n", " "), { x: x + 0.32, y: yy, w: cw - 0.6, h: mh, valign: "middle", fontFace: FACE, fontSize: 13, bold: true, color: WHITE, margin: 0 });
      yy += mh + 0.18;
    });
    if (i < cols.length - 1) {
      s.addText("→", { x: x + cw - 0.04, y: y0 + 1.8, w: cg + 0.08, h: 0.6, align: "center", valign: "middle", fontFace: FACE, fontSize: 24, bold: true, color: ACCENT, margin: 0 });
    }
  });
  s.addText("拿掉標籤 → 拿掉 memory bank → 拿掉負樣本 → 換新架構", {
    x: 0.9, y: 6.55, w: 11.5, h: 0.45, align: "center", fontFace: MONO, fontSize: 13, bold: true, color: ICE, margin: 0 });
  s.addNotes(NOTES.evoMap);
}

// ============================================================
// ACT 1
// ============================================================
actDivider("1", E1, "2018 – 2019", "起點：沒有標籤，\n要怎麼定義「任務」？", NOTES.act1);

// --- Instance Discrimination ---
methodSlide({
  era: A1("ERA 1 · 2018"), name: "Instance Discrimination", demo: "demo_assets/gifs/instance_discrimination.gif", venue: "CVPR 2018",
  authors: "Z. Wu, Y. Xiong, S. X. Yu, D. Lin — Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
  idea: "把每張圖當成「自己的一類」，做 instance-level 區分，完全不需人工標籤。",
  mechanism: ["Memory Bank 儲存每張圖的 L2 特徵向量", "NCE 近似完整 softmax（常數 Z 首批估計後固定）", "弱增強、n_views=1，bank 提供「第二視角」"],
  contribution: "非參數記憶庫先驅；痛點：bank 特徵會過時。",
  diagram: (s, o) => siamese(s, o, { single: true, proj: false, top: "編碼器\nf", loss: "NCE", extra: "Memory\nBank（全資料特徵）" }),
  loss: "P(i|v) = exp(vᵢᵀf/τ) / Σⱼ exp(vⱼᵀf/τ)", lossNote: "非參數 softmax + NCE 近似",
});

// --- Invariant Spread ---
methodSlide({
  era: A1("ERA 1 · 2019"), name: "Invariant and Spreading Instance Feature", demo: "demo_assets/gifs/invariant_spread.gif", venue: "CVPR 2019",
  authors: "M. Ye, X. Zhang, P. C. Yuen, S.-F. Chang — Unsupervised Embedding Learning via Invariant and Spreading Instance Feature",
  idea: "不用記憶庫，直接在 batch 內把同圖兩視角拉近、不同圖推開。",
  mechanism: ["兩個增強視角 + 對稱 InfoNCE", "in-batch 負樣本（無 queue、無 memory bank）", "SimCLR 的直系祖先"],
  contribution: "in-batch softmax 對比的雛形，啟發後續整個 Era 2。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: false, loss: "對稱\nInfoNCE" }),
  loss: "−Σ log exp(sim/τ) / Σ exp(sim/τ)", lossNote: "in-batch 對稱 softmax",
});

// ============================================================
// ACT 2
// ============================================================
actDivider("2", E2, "2020", "大爆發：\n負樣本要從哪裡來？", NOTES.act2);

// --- MoCo v1 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v1", demo: "demo_assets/gifs/moco_v1.gif", venue: "CVPR 2020",
  authors: "K. He, H. Fan, Y. Wu, S. Xie, R. Girshick — Momentum Contrast for Unsupervised Visual Representation Learning",
  idea: "用動量編碼器 + FIFO queue，提供大量「不過時」的負樣本。",
  mechanism: ["query 編碼器（梯度更新）+ key 編碼器（EMA 動量）", "FIFO queue 儲存歷史 key 當負樣本", "負樣本量與 batch size 解耦"],
  contribution: "小 batch 也能有大量、一致的負樣本。",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(梯度)", bot: "f_k", botNote: "EMA 動量", proj: false, loss: "InfoNCE", extra: "Queue\n(歷史 key 負樣本)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "queue 提供大量負樣本",
});

// --- MoCo v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "MoCo v2", demo: "demo_assets/gifs/moco_v2.gif", venue: "arXiv 2020",
  authors: "X. Chen, H. Fan, R. Girshick, K. He — Improved Baselines with Momentum Contrastive Learning",
  idea: "把 SimCLR 的工程改良搬進 MoCo，用更少資源逼近其表現。",
  mechanism: ["2 層 MLP 投影頭（取代 v1 的線性層）", "加入 Gaussian blur 強增強 + cosine LR", "其餘架構與 v1 完全相同"],
  contribution: "MLP 頭 + 強增強 = 小資源逼近 SimCLR。",
  diagram: (s, o) => siamese(s, o, { top: "f_q\n(梯度)", bot: "f_k", botNote: "EMA 動量", proj: true, loss: "InfoNCE", extra: "Queue\n(歷史 key 負樣本)" }),
  loss: "−log exp(q·k₊/τ) / Σ_{queue} exp(q·k/τ)", lossNote: "同 v1，新增 MLP 投影頭 g",
});

// --- SimCLR v1 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SimCLR v1", demo: "demo_assets/gifs/simclr_v1.gif", venue: "ICML 2020",
  authors: "T. Chen, S. Kornblith, M. Norouzi, G. Hinton — A Simple Framework for Contrastive Learning of Visual Representations",
  idea: "不要 queue/bank，直接用超大 batch，batch 內互當負樣本。",
  mechanism: ["共享 backbone + MLP 投影頭", "對稱 NT-Xent；loss 算在 z、評估用 h", "強增強（color jitter s=1.0 + blur）是關鍵"],
  contribution: "證明「簡單框架 + 強增強」即可；痛點：需大 batch。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "NT-Xent" }),
  loss: "−log exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(·)", lossNote: "NT-Xent（in-batch 對稱）",
});

// --- SimCLR v2 ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SimCLR v2", demo: "demo_assets/gifs/simclr_v2.gif", venue: "NeurIPS 2020",
  authors: "T. Chen, S. Kornblith, K. Swersky, M. Norouzi, G. Hinton — Big Self-Supervised Models are Strong Semi-Supervised Learners",
  idea: "更深的投影頭 + 更大模型 → 強半監督學習者。",
  mechanism: ["3 層投影頭（僅預訓練階段使用）", "更大、更深的 backbone", "蒸餾到小模型完成半監督"],
  contribution: "大型 SSL 模型是強半監督學習者（pretrain→distill）。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "NT-Xent" }),
  loss: "−log exp(sim(zᵢ,zⱼ)/τ) / Σ_{k≠i} exp(·)", lossNote: "同 v1，投影頭加深為 3 層",
});

// --- SwAV ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "SwAV", demo: "demo_assets/gifs/swav.gif", venue: "NeurIPS 2020",
  authors: "M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, A. Joulin — Unsupervised Learning of Visual Features by Contrasting Cluster Assignments",
  idea: "不再兩兩比較，改成線上分群（prototypes）。",
  mechanism: ["multi-crop：2 大 + N 小裁切", "Sinkhorn-Knopp 最佳傳輸求軟分配 code", "swapped prediction：互相預測對方的 code"],
  contribution: "免成對負樣本的對比；多裁切大幅提升效率。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "Swapped\nCE", extra: "Prototypes C\n+ Sinkhorn" }),
  loss: "−Σ qₜ·log pₛ − qₛ·log pₜ", lossNote: "code q 由 Sinkhorn-Knopp 求得",
});

// --- InfoMin ---
methodSlide({
  era: A2("ERA 2 · 2020"), name: "InfoMin", demo: "demo_assets/gifs/infomin.gif", venue: "NeurIPS 2020",
  authors: "Y. Tian, C. Sun, B. Poole, D. Krishnan, C. Schmid, P. Isola — What Makes for Good Views for Contrastive Learning?",
  idea: "退一步問：什麼才是「好的」增強視角？",
  mechanism: ["沿用 SimCLR 的 backbone 與 NT-Xent loss", "更激進增強（s=1.5、grayscale 0.4、去 blur）", "去除紋理／顏色等捷徑相關性"],
  contribution: "minimal sufficient 視角原則：共享任務資訊、其餘越少越好。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "NT-Xent" }),
  loss: "min I(v₁;v₂)  +  NT-Xent(z₁,z₂)", lossNote: "loss 同 SimCLR，重點在視角設計",
});

// ============================================================
// ACT 3
// ============================================================
actDivider("3", E3, "2020 – 2021", "異端：\n我們真的需要負樣本嗎？", NOTES.act3);

// --- BYOL ---
methodSlide({
  era: A3("ERA 3 · 2020"), name: "BYOL", demo: "demo_assets/gifs/byol.gif", venue: "NeurIPS 2020",
  authors: "J.-B. Grill, F. Strub, F. Altché, et al. (DeepMind) — Bootstrap Your Own Latent",
  idea: "完全不用負樣本也能學 —— 靠 predictor 不對稱 + EMA target。",
  mechanism: ["online 分支多一個 predictor（製造不對稱）", "target 分支 = online 的 EMA（動量 0.996→1）", "預測 target 表徵，target 端 stop-gradient"],
  contribution: "證明 predictor 不對稱即可防坍塌，震撼全場。",
  diagram: (s, o) => siamese(s, o, { top: "online\nf_θ", bot: "target\nf_ξ", botNote: "EMA", proj: true, pred: true, sgBot: true, loss: "MSE" }),
  loss: "‖ q(z₁) − sg(z₂′) ‖²  (對稱化)", lossNote: "predictor 不對稱 + sg(target)",
});

// --- SimSiam ---
methodSlide({
  era: A3("ERA 3 · 2021"), name: "SimSiam", demo: "demo_assets/gifs/simsiam.gif", venue: "CVPR 2021",
  authors: "X. Chen, K. He — Exploring Simple Siamese Representation Learning",
  idea: "連 EMA 都不要，只靠一個 stop-gradient 就不坍塌。",
  mechanism: ["共享 backbone（無動量編碼器、無 queue）", "online 分支 predictor + target 端 stop-gradient", "負 cosine 相似度損失"],
  contribution: "最小化證明：stop-gradient 才是防坍塌的關鍵。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, pred: true, sgBot: true, loss: "−cos" }),
  loss: "½D(p₁, sg z₂) + ½D(p₂, sg z₁)", lossNote: "stop-grad 是唯一防坍塌機制",
});

// --- Barlow Twins ---
methodSlide({
  era: A3("ERA 3 · 2021"), name: "Barlow Twins", demo: "demo_assets/gifs/barlow_twins.gif", venue: "ICML 2021",
  authors: "J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny — Self-Supervised Learning via Redundancy Reduction",
  idea: "換個哲學：讓兩視角嵌入的互相關矩陣趨近單位矩陣。",
  mechanism: ["計算兩視角嵌入的互相關矩陣 C", "對角→1（不變性）、非對角→0（去冗餘）", "高維投影頭（8192）效果最好"],
  contribution: "無負樣本／EMA／predictor，全靠 loss 作用於 C。",
  diagram: (s, o) => siamese(s, o, { top: "編碼器\nf", bot: "編碼器\nf", botNote: "共享權重", proj: true, loss: "Cross-Corr\nC → I" }),
  loss: "Σᵢ(1−Cᵢᵢ)² + λ Σ_{i≠j} Cᵢⱼ²", lossNote: "互相關矩陣 → 單位矩陣（去冗餘）",
});

// ============================================================
// ACT 4
// ============================================================
actDivider("4", E4, "2021 → 今天", "架構遷移：\n換上 Transformer", NOTES.act4);

// --- MoCo v3 ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "MoCo v3", demo: "demo_assets/gifs/moco_v3.gif", venue: "ICCV 2021",
  authors: "X. Chen, S. Xie, K. He — An Empirical Study of Training Self-Supervised Vision Transformers",
  idea: "把對比學習搬上 ViT，並找出讓它穩定訓練的配方。",
  mechanism: ["凍結 patch embedding 投影（最關鍵的穩定性修正）", "對稱 in-batch InfoNCE（丟掉 queue）", "AdamW + cosine LR（非 SGD/LARS），m=0.99"],
  contribution: "ViT 對比訓練的穩定配方；大 batch 下毋需 queue。",
  diagram: (s, o) => siamese(s, o, { top: "ViT f_q\n(梯度)", bot: "ViT f_k", botNote: "EMA 動量", proj: true, loss: "對稱\nInfoNCE" }),
  loss: "ctr(q₁,k₂) + ctr(q₂,k₁)", lossNote: "對稱 InfoNCE，無 queue",
});

// --- DINO ---
methodSlide({
  era: A4("ERA 4 · 2021"), name: "DINO", demo: "demo_assets/gifs/dino.gif", venue: "ICCV 2021",
  authors: "M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin — Emerging Properties in Self-Supervised ViTs",
  idea: "student–teacher 自蒸餾，無對比負樣本。",
  mechanism: ["teacher = student 的 EMA（只看 global crop）", "teacher 輸出做 centering + sharpening", "cross-entropy：student 預測 teacher 分布"],
  contribution: "attention map 浮現語意分割；防坍塌靠 centering+sharpening。",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "Cross-\nEntropy", extra: "centering\n+ sharpening" }),
  loss: "−Σ Pₜ log Pₛ , Pₜ=σ((g−C)/τₜ)", lossNote: "σ=softmax；centering + sharpening 防坍塌",
});

// --- DINOv2 ---
methodSlide({
  era: A4("ERA 4 · 2023"), name: "DINOv2", demo: "demo_assets/methods/dinov2.png", venue: "TMLR 2024",
  authors: "M. Oquab, T. Darcet, T. Moutakanni, et al. (Meta AI) — Learning Robust Visual Features without Supervision",
  idea: "把自蒸餾規模化，做出通用視覺「基礎模型」。",
  mechanism: ["DINO（影像級）+ iBOT（patch 級）自蒸餾", "LVD-142M 大規模精選資料 + 訓練技巧", "本教學僅提供特徵抽取／微調 demo"],
  contribution: "免微調即強的通用特徵，foundation model 的前身。",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "DINO\n+ iBOT", extra: "iBOT\n(patch 級遮罩)" }),
  loss: "ℒ_DINO + ℒ_iBOT + 正則項", lossNote: "影像級 + patch 級自蒸餾，規模化",
});

// --- DINOv3 (2025 續章) ---
methodSlide({
  era: A4("ERA 4 · 2025"), name: "DINOv3", demo: "demo_assets/methods/dinov3.png", venue: "arXiv 2025",
  authors: "O. Siméoni, H. V. Vo, et al. (Meta AI) — DINOv3",
  idea: "同一條 DINO 路線再往上推到極限規模，並用 Gram anchoring 穩住 dense 特徵。",
  mechanism: ["沿用 DINO + iBOT 自蒸餾，放大到 7B 參數 / 17 億張影像", "Gram anchoring：訓練後期穩住 patch 級 dense 特徵、防退化", "再從巨模型蒸餾出 ViT-S/B/L 等好用的小模型"],
  contribution: "凍結 backbone 的 dense 特徵達 SOTA；免微調就打贏專用模型。",
  diagram: (s, o) => siamese(s, o, { top: "student\nf_s", bot: "teacher\nf_ξ", botNote: "EMA", proj: true, sgBot: true, loss: "DINO+iBOT\n+Gram", extra: "Gram\nanchoring" }),
  loss: "ℒ_DINO + ℒ_iBOT + ℒ_Gram", lossNote: "自蒸餾 + Gram anchoring 穩住 dense 特徵",
});

// 15. Comparison / collapse payoff table
{
  const s = pres.addSlide();
  contentHeader(s, "收束", "0E7490", "回頭看：每一代怎麼防坍塌？");
  const hdr = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: FACE, fontSize: 14, align: "center", valign: "middle" } });
  const rows = [
    [hdr("時代"), hdr("代表方法"), hdr("用負樣本？"), hdr("防坍塌機制")],
    ["Era 1 · 2018", "Instance Discrimination", { text: "是（memory bank）", options: { color: "2A8F5F" } }, "用大量負樣本把表徵推開"],
    ["Era 2 · 2020", "MoCo / SimCLR", { text: "是（queue / batch）", options: { color: "2A8F5F" } }, "in-batch / queue 負樣本"],
    ["Era 2 · 2020", "SwAV", { text: "否（用 prototype）", options: { color: E3, bold: true } }, "線上分群 + Sinkhorn 均勻分配"],
    ["Era 3 · 2021", "BYOL / SimSiam", { text: "否", options: { color: E3, bold: true } }, "EMA / stop-gradient 打破對稱"],
    ["Era 3 · 2021", "Barlow Twins", { text: "否", options: { color: E3, bold: true } }, "冗餘消除（特徵去相關）"],
    ["Era 4 · 2021+", "DINO", { text: "否", options: { color: E3, bold: true } }, "centering + sharpening"],
  ];
  s.addTable(rows, {
    x: 0.7, y: 2.3, w: 11.93, colW: [2.1, 3.5, 2.73, 3.6],
    rowH: [0.5, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52],
    border: { pt: 0.5, color: "D5DCEA" }, align: "left", valign: "middle",
    fontFace: FACE, fontSize: 13, color: INK, margin: [0, 0.12, 0, 0.12],
    fill: { color: CARD },
  });
  s.addText([
    { text: "Punchline：14 個方法，其實都在回答同一個問題 —— ", options: { color: INK } },
    { text: "「沒有標籤，怎麼不讓表徵坍塌？」", options: { bold: true, color: "0E7490" } },
  ], { x: 0.7, y: 6.35, w: 11.9, h: 0.5, align: "center", fontFace: FACE, fontSize: 15, italic: true, margin: 0 });
  s.addNotes(NOTES.collapseTable);
}

// 15b. Epoch progression (appendix) — clusters emerging over training, all 7 self-trained methods
function progSlide(suffix, rows, notes) {
  const s = pres.addSlide();
  contentHeader(s, "訓練過程", "0E7490", "分群如何隨 epoch 浮現？" + suffix);
  s.addText("自訓 SSL（CIFAR-10，ResNet-18 / ViT-S，2×H100，200 epoch）：特徵從 epoch 0 的「一團」逐步分化成清楚的類別群落。", {
    x: 0.7, y: 1.92, w: 11.9, h: 0.4, fontFace: FACE, fontSize: 12.5, color: MUTE, margin: 0 });
  const n = rows.length;
  const top = 2.34, bottom = 6.84, gap = 0.12;
  const sh = (bottom - top - (n - 1) * gap) / n;
  const sw = 9.6, sx = 3.0;
  let yy = top;
  rows.forEach((r) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: yy, w: 2.1, h: sh, fill: { color: NAVY } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: yy, w: 0.09, h: sh, fill: { color: r.c } });
    s.addText(r.label, { x: 0.95, y: yy + sh / 2 - 0.32, w: 1.95, h: 0.34, fontFace: FACE, fontSize: 14, bold: true, color: WHITE, margin: 0 });
    s.addText(r.sub, { x: 0.95, y: yy + sh / 2 + 0.03, w: 1.95, h: 0.3, fontFace: FACE, fontSize: 10, color: ICE, margin: 0 });
    const p = "demo_assets/progression/" + r.key + ".png";
    if (fs.existsSync(p)) s.addImage({ path: p, x: sx, y: yy, w: sw, h: sh, sizing: { type: "contain", w: sw, h: sh } });
    yy += sh + gap;
  });
  s.addText("← 隨機初始化（epoch 0）　　訓練越久，同類越聚、異類越分　　充分訓練（epoch 200）→", {
    x: 3.0, y: 6.9, w: 9.6, h: 0.3, align: "center", fontFace: FACE, fontSize: 10, italic: true, color: MUTE, margin: 0 });
  if (notes) s.addNotes(notes);
}
progSlide("（1/3 · Era 1–2）", [
  { key: "instance_discrimination", label: "Instance Discrim.", sub: "memory bank", c: E1 },
  { key: "invariant_spread", label: "Invariant Spread", sub: "in-batch softmax", c: E1 },
  { key: "simclr_v1", label: "SimCLR v1", sub: "in-batch 對比", c: E2 },
  { key: "simclr_v2", label: "SimCLR v2", sub: "深 3 層投影頭", c: E2 },
], NOTES.prog1);
progSlide("（2/3 · Era 2–3）", [
  { key: "infomin", label: "InfoMin", sub: "視角設計", c: E2 },
  { key: "byol", label: "BYOL", sub: "predictor + EMA", c: E3 },
  { key: "simsiam", label: "SimSiam", sub: "stop-gradient", c: E3 },
], NOTES.prog2);
progSlide("（3/3 · 官方權重方法自訓補完）", [
  { key: "moco_v1", label: "MoCo v1", sub: "queue + 動量", c: E2 },
  { key: "moco_v2", label: "MoCo v2", sub: "MLP 頭 + 強增強", c: E2 },
  { key: "swav", label: "SwAV", sub: "prototype + Sinkhorn", c: E2 },
  { key: "barlow_twins", label: "Barlow Twins", sub: "去冗餘", c: E3 },
  { key: "moco_v3", label: "MoCo v3", sub: "ViT + 凍結 patch", c: E4 },
  { key: "dino", label: "DINO", sub: "自蒸餾 + centering", c: E4 },
], NOTES.prog3);

// 16. Live demo
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: PH, fill: { color: ACCENT } });
  s.addText("LIVE DEMO", { x: 0.9, y: 0.5, w: 6, h: 0.45, fontFace: MONO, fontSize: 15, bold: true, color: ACCENT, charSpacing: 4, margin: 0 });
  s.addText("眼見為憑：特徵空間從「一團」變「分群」", { x: 0.9, y: 0.95, w: 11.5, h: 0.65, fontFace: FACE, fontSize: 26, bold: true, color: WHITE, margin: 0 });

  // command strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 1.75, w: 11.5, h: 0.98, fill: { color: "060B16" }, line: { color: NAVY3, width: 1 }, shadow: makeShadow() });
  s.addText([
    { text: "$ python train.py --config configs/simclr_v1_resnet18.yaml --data-dir data/cifar10", options: { breakLine: true, color: "7CF0A0" } },
    { text: "$ python eval/umap_vis.py configs/simclr_v1_resnet18.yaml --ckpt checkpoints/last.ckpt", options: { color: "7CF0A0" } },
  ], { x: 1.15, y: 1.9, w: 11, h: 0.7, fontFace: MONO, fontSize: 12.5, lineSpacingMultiple: 1.3, margin: 0 });

  // before / after UMAP images
  const imgY = 3.25, imgS = 2.65, labY = 2.88;
  const beforeX = 3.0, afterX = 7.68; // arrow sits in the ~0.95" gap between
  s.addText("訓練前 · 隨機初始化", { x: beforeX, y: labY, w: imgS, h: 0.34, align: "center", fontFace: FACE, fontSize: 14, bold: true, color: "FF8FA8", margin: 0 });
  s.addText("訓練後 · 預訓練特徵", { x: afterX, y: labY, w: imgS, h: 0.34, align: "center", fontFace: FACE, fontSize: 14, bold: true, color: "7CF0A0", margin: 0 });
  s.addImage({ path: "demo_assets/umap_before.png", x: beforeX, y: imgY, w: imgS, h: imgS, sizing: { type: "contain", w: imgS, h: imgS } });
  s.addImage({ path: "demo_assets/umap_after.png", x: afterX, y: imgY, w: imgS, h: imgS, sizing: { type: "contain", w: imgS, h: imgS } });
  s.addShape(pres.shapes.RECTANGLE, { x: beforeX, y: imgY, w: imgS, h: imgS, fill: { color: "FFFFFF", transparency: 100 }, line: { color: "C0395A", width: 1.5 } });
  s.addShape(pres.shapes.RECTANGLE, { x: afterX, y: imgY, w: imgS, h: imgS, fill: { color: "FFFFFF", transparency: 100 }, line: { color: "2A8F5F", width: 1.5 } });
  s.addText("→", { x: beforeX + imgS, y: imgY + imgS / 2 - 0.4, w: afterX - beforeX - imgS, h: 0.8, align: "center", valign: "middle", fontFace: FACE, fontSize: 40, bold: true, color: ACCENT, margin: 0 });

  s.addText([
    { text: "※ 教學示意：以「ImageNet 預訓練 backbone」代理訓練後特徵（CIFAR-10）。", options: { breakLine: true } },
    { text: "  換上你自己訓練好的 SSL checkpoint 即可重現真實結果。", options: {} },
  ], { x: 0.9, y: 6.35, w: 11.5, h: 0.8, align: "center", fontFace: FACE, fontSize: 12, italic: true, color: MUTE, margin: 0, lineSpacingMultiple: 1.15 });
  s.addNotes(NOTES.liveDemo);
}

// 17. Closing
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.OVAL, { x: -2.5, y: 4.5, w: 6, h: 6, fill: { color: NAVY3, transparency: 45 } });
  s.addText("演化史，一句話總結", { x: 0.9, y: 0.85, w: 11.5, h: 0.6, fontFace: FACE, fontSize: 26, bold: true, color: ACCENT, margin: 0 });
  s.addText([
    { text: "5 年來，對比學習不斷「拿掉依賴」——\n從 memory bank，到負樣本，最後連", options: { color: WHITE } },
    { text: "架構", options: { color: ACCENT, bold: true } },
    { text: "都換新。", options: { color: WHITE } },
  ], { x: 0.9, y: 1.6, w: 11.5, h: 1.5, fontFace: FACE, fontSize: 28, bold: true, lineSpacingMultiple: 1.15, margin: 0 });

  const takeaways = [
    "所有方法共用 InfoNCE 直覺：拉近正樣本、推開負樣本",
    "真正的難題是「防坍塌」，負樣本只是其中一條路",
    "趨勢：更少假設、更大規模 → 走向 foundation model",
  ];
  let yy = 3.5;
  takeaways.forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: 0.95, y: yy + 0.02, w: 0.42, h: 0.42, fill: { color: ACCENT } });
    s.addText(String(i + 1), { x: 0.95, y: yy + 0.02, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: FACE, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(t, { x: 1.6, y: yy, w: 11, h: 0.46, valign: "middle", fontFace: FACE, fontSize: 16.5, color: ICE, margin: 0 });
    yy += 0.62;
  });

  s.addShape(pres.shapes.LINE, { x: 0.9, y: 5.75, w: 11.5, h: 0, line: { color: NAVY3, width: 1 } });
  s.addText([
    { text: "開源教學專案 · 14 種方法統一實作 · 每個 loss 都能獨立閱讀", options: { color: ICE } },
  ], { x: 0.9, y: 5.95, w: 11.5, h: 0.5, fontFace: FACE, fontSize: 14, margin: 0 });
  s.addText("謝謝聆聽 — 歡迎提問", { x: 0.9, y: 6.5, w: 11.5, h: 0.6, fontFace: FACE, fontSize: 22, bold: true, color: WHITE, margin: 0 });
  s.addNotes(NOTES.closing);
}

pres.writeFile({ fileName: "contrastive_learning_evolution.pptx" }).then((f) => console.log("WROTE", f));
