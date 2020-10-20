# coding:utf-8
from tqdm import tqdm


class MatchFinder:
    def __init__(self, R_corpus, C_corpus):
        self.R_corpus = R_corpus
        self.C_corpus = C_corpus
        self.R = None
        self.C = None
        self.keys = []
        self.matches = []

    def corres(self, iR=0, iC=0, H=set()):
        if iR >= len(self.R):
            self.keys.append(H)
            self.matches.append((self.R, self.C))
            return True

        if iC >= len(self.C):
            return False

        if self.R.count(self.R[iR]) <= 1:
            return self.corres(iR + 1, iC, H)

        if self.C.count(self.C[iC]) <= 1:
            return self.corres(iR, iC + 1, H)

        RestR = len([i for i in self.R[iR:] if self.R.count(i) > 1])
        RestC = len([i for i in self.C[iC:] if self.C.count(i) > 1])

        C_values = [e[1] for e in H if e[0] == R[iR]]

        if C_values:
            if self.C[iC] in C_values:
                if RestR > RestC:
                    return False
                if self.corres(iR + 1, iC + 1, H):
                    return True
            else:
                return False
        else:
            if RestR <= RestC:
                H2 = H.copy()
                H2.add((self.R[iR], self.C[iC]))
                if self.corres(iR + 1, iC + 1, H2):
                    return True

        return self.corres(iR, iC + 1, H)

    def find_matches(self):
        for i in tqdm(range(0, len(self.R_corpus) - 20, 20)):
            self.R = self.R_corpus[i:i+20]
            for j in range(0, len(self.C_corpus) - 30, 30):
                self.C = self.C_corpus[j:j+30]
                if self.corres():
                    self.matches.append((self.R, self.C))


R = list("新しい記事を書こうという気持ちになるまで長い時間がかかった書きたいことはたくさんあったけれど息子を産んだ後は書く時間があまりなかった幸運にも息子はこの四月から保育園に入ることができ私はまた働き始めた日本では近頃多くの人が保育園問題について話している特に東京では十分な施設がないので子どもを保育園に入れることがとても大変だ今私は東京に住んでいるので息子を保育園に入れるのは不可能だろうと思っていたしかし驚いたことに息子は受け入れてもらえた息子を産んだ後に気がついたことの一つに日本人は赤ちゃんを目にしたときは全然シャイではないということがある例えば私が息子と出掛けたときたくさんの人が私に話しかけ息子ににっこりと微笑んだりするこういうことが何度も起きたのでだれが日本人はシャイだなんて言ったのという気持ちになってしまった子どもを持ったおかげで以前は知らなかった日本社会と日本人のもう一つの面を発見した今後の記事でそれについて書いていければと思う")

C = list("アタラシイキジヲカコウトイウキモチニナルマデナガイジカンガカカッタカキタイコトハタクサンアッタケレドムスコヲウンダノチハカクジカンガアマリナカッタコウウンニモムスコハコノシガツカラホイクエンニハイルコトガデキワタシハマタハタラキハジメタニッポンデハチカゴロオオクノヒトガホイクエンモンダイニツイテハナシテイルトクニトウキョウデハジュウブンナシセツガナイノデコドモヲホイクエンニイレルコトガトテモタイヘンダイマワタシハトウキョウニスンデイルノデムスコヲホイクエンニイレルノハフカノウダロウトオモッテイタシカシオドロイタコトニムスコハウケイレテモラエタムスコヲウンダノチニキガツイタコトノヒトツニニッポンジンハアカチャンヲメニシタトキハゼンゼンシャイデハナイトイウコトガアルタトエバワタシガムスコトデカケタトキタクサンノヒトガワタシニハナシカケムスコニニッコリトホホエンダリスルコウイウコトガナンドモオキタノデダレガニッポンジンハシャイダナンテイッタノトイウキモチニナッテシマッタコドモヲモッタオカゲデイゼンハシラナカッタニッポンシャカイトニッポンジンノモウヒトツノメンヲハッケンシタコンゴノキジデソレニツイテカイテイケレバトオモウ")

mf = MatchFinder(R, C)
mf.R = "後の記事でそれについて書いていければと思"
mf.C = "ノメンヲハッケンシタコンゴノキジデソレニツイテカイテイケレバ"
mf.corres()

for m in mf.matches:
    print(''.join(m[0]))
    print(''.join(m[1]))
    print('')

print(mf.keys)