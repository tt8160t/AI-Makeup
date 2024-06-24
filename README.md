# AI-Makeup
## 目的
- 透過即時連線拍照的方式，讓使用者可以透過AI上妝模型去進行建議上妝。

## 使用的語言及工具
### 語言
- JAVA EE(Servlet+JSP)
- Python
- Tensorflow
- MySQL
- HTML, CSS, JavaScript

### 工具
- Eclipse
- VsCode
- git
- MySQL Workbench

## 簡易流程
1. JAVA前後端的建立以及串接資料庫，透過MVC的架構實現，會員登入及登出、創建會員、顯示會員資訊等等
2. 將前端頁面JSP與後端Flask串接達到即時拍照的功能
3. 拍完的照片進行上妝後，透過http request的協定方式傳到後端python中的程式碼，進行跑模上妝，並回傳。

## 流程中碰到的問題
1. 無法呈現即時上妝，因為消耗的資源過大導致類格很嚴重，因此後續是透過"即時拍照"後才進行上妝
2. 傳達到後端之後，對於圖片的一些處理(包含切割、跑模、回傳圖片等等流程)，由於隊員當初都是一個功能一個功能輸出一張圖片，上一個功能輸出的圖片傳給下一個，導致很多呼叫function都是讀取上一個功能的輸出圖片所寫成的絕對路徑。

## 備註
JAVA EE的思考流程: 
第一次Run時會先初始化init中的initalize.java，因為她有設定contextInitialized(這是一個JAVA EE的listener他在
初始化時會先跑這個程式)，並且老師裡面有寫好，將東西匯入資料庫，跟一些後續會較常用的“路徑”
較值得注意的是它裡面會再去連資料庫，透過META-INF中的context.xml。PS: 在WEB-INF中的lib裡面有些jar檔也是要額外安裝，譬如跟mysql連線之類的

1. ToLogin.java
2. 透過自定義的PathConverter可將網址上目前假如是…/login自動去找在WEB-IN中的login.jsp
3. 填好帳號密碼後按下登入會跳到Login.java(/Login.do)，裡面會呼叫很多MemberDAO裡面的方法，並將得到的資料都存在session中
4. 後續在每個jsp上如果有需要都可以讀取這些session中的資訊

## PS:
在註冊會員時會碰到的問題，因為我們會同時上傳一班資料跟圖片，而圖片由於過大所以在jsp的form表單中新增
enctype=“multipart/form-data”，而這也會導致我們後續使用getParameter會讀不到資料，因為他已經被封裝起來了
接下來就得使用getValue(request.getPart(“email”));並且宣告@MultipartConfig(
fileSizeThreshold = 1024 * 1024 * 1, // 1 MB
maxFileSize = 1024 * 1024 * 10, // 10 MB
maxRequestSize = 1024 * 1024 * 15 // 15 MB
)








