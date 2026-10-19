# La xarxa MNIST

## El problema: classificar imatges

**MNIST** és un conjunt d'imatges de dígits escrits a mà. Cada imatge té 28 × 28 píxels i una etiqueta correcta entre 0 i 9. L'objectiu és construir un model que, a partir dels píxels, predigui quin dígit hi apareix.

En aquesta pràctica, cada imatge es converteix en un vector de 784 valors. Els valors dels píxels es normalitzen dividint-los per 255, de manera que queden entre 0 i 1. El model rep els píxels com una llista, sense conservar explícitament l'estructura de files i columnes.

## Neurones, pesos i capes

Una neurona rep uns valors d'entrada, els combina amb **pesos** i un **biaix** (*bias*), i produeix una sortida. De manera simplificada:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Els pesos determinen la importància de cada entrada. El biaix permet desplaçar el resultat. Una **capa** agrupa neurones; una xarxa neuronal connecta capes perquè la sortida d'una serveixi d'entrada a la següent.

L'arquitectura de la pràctica té 784 entrades, una capa oculta de 64 neurones i 10 sortides, una per cada dígit:

`784 → 64 → 10`

## Funció d'activació: ReLU

Les funcions d'activació permeten a la xarxa aprendre patrons més complexos que una simple transformació lineal. La pràctica utilitza **ReLU** (*Rectified Linear Unit*):

$$
\operatorname{ReLU}(z) = \max(0, z)
$$

Els valors negatius passen a ser zero i els positius es mantenen. La xarxa de la pràctica és `Linear(784, 64) → ReLU → Linear(64, 10)`.

## Sortida de la xarxa: logits i predicció

Les dades travessen les capes en el pas **forward**. La xarxa produeix 10 valors de sortida per imatge, anomenats **logits**. El logit més alt indica quin dígit prediu el model. Els logits encara no són probabilitats; la funció de pèrdua els pot utilitzar directament.

S’anomenen **logits** perquè són els valors «crus» que la xarxa calcula just abans de convertir-los en probabilitats. Per a diverses classes, com els deu dígits de MNIST, s’anomena habitualment *logits* als valors d’entrada de la funció `softmax`, que els converteix en probabilitats.

Per exemple, les sortides `[2.0, 1.0, -1.0]` són logits: poden ser negatius i no han de sumar 1. Després de `softmax`, esdevenen probabilitats que sí que sumen 1. La classe amb el logit més alt també tindrà la probabilitat més alta.
# L'entrenament

## Dades d'entrenament i de validació

Les dades es divideixen segons l'ús que se'n fa:

- **Entrenament:** el model les utilitza per calcular gradients i modificar els pesos.
- **Validació:** no s'utilitza per actualitzar els pesos. Serveix per comprovar com respon el model a exemples que no ha fet servir per aprendre.

Mesurar el rendiment només amb dades d'entrenament pot donar una impressió massa optimista. La validació ajuda a veure si el model aprèn patrons que generalitzen a exemples no vistos.

## Per què fem servir batchs?

En lloc de passar totes les imatges d'entrenament a la xarxa alhora, les dividim en grups més petits anomenats **batchs**. Això limita la memòria necessària per calcular les prediccions i els gradients. També permet actualitzar els pesos diverses vegades durant una època: després de cada batch, el model ja pot fer una petita correcció abans de processar el grup següent.

Cada batch dona una estimació de com cal canviar els pesos a partir dels exemples que conté. Un batch petit necessita menys memòria i fa més actualitzacions per època, però les estimacions poden variar més d'un grup a l'altre. Un batch gran dona una estimació més estable, però necessita més memòria i permet menys actualitzacions per època. Per això la mida del batch és un compromís, no hi ha un valor que sigui sempre millor.

## Cicle d'aprenentatge: quatre passos per batch

El cicle es repeteix per a cada *batch* (grup d'imatges). El model comença amb pesos inicials i els va ajustant a partir dels exemples:

1. **Forward: fer prediccions.** El batch passa per les capes i, per a cada imatge, la xarxa produeix 10 logits. El logit més alt correspon a la predicció. Encara no s'han canviat els pesos.

2. **Loss: mesurar l'error.** Es comparen les prediccions amb les etiquetes correctes del batch. La pràctica utilitza **cross-entropy**, que penalitza especialment les prediccions incorrectes fetes amb molta confiança. Una loss més baixa indica prediccions millors segons aquesta mesura. Al codi, `F.cross_entropy(model(Xt[idx]), yt[idx])` calcula el forward i la loss. No cal aplicar softmax abans: `cross_entropy` rep els logits directament.

3. **Backward: calcular com corregir-se.** `loss.backward()` propaga l'error cap enrere per la xarxa i calcula un **gradient** per a cada pes i biaix. El gradient indica com canviaria la loss si canviéssim lleugerament aquell paràmetre. Aquest pas calcula informació per a la correcció, però encara no modifica els paràmetres.

4. **Step: actualitzar els paràmetres.** `opt.step()` utilitza els gradients per modificar els pesos i els biaixos, en una direcció que tendeix a reduir la loss. La pràctica utilitza **descens del gradient estocàstic** (*SGD*). De manera simplificada:

$$
w_{\mathrm{nou}} = w_{\mathrm{actual}} - \eta \cdot g
$$

Aquí, $\eta$ és el learning rate i $g$ és el gradient del pes.

Abans de calcular gradients nous, `opt.zero_grad()` esborra els gradients acumulats del pas anterior. PyTorch els acumula per defecte, així que no esborrar-los barrejaria els gradients de batches diferents. Al codi, l'ordre és: calcular la loss, esborrar gradients antics, calcular els nous gradients i actualitzar els paràmetres. Després, el batch següent ja utilitza els pesos actualitzats.

## Batch, actualització i època

- **Batch:** grup d'exemples processats junts per calcular una loss i els gradients.
- **Actualització:** una crida a `opt.step()`. En aquesta pràctica n'hi ha una per batch.
- **Època:** una passada completa per totes les dades d'entrenament seleccionades.

Si hi ha 10.000 exemples i `batch=100`, hi ha 100 batches i 100 actualitzacions per època. Amb 3 èpoques hi ha 300 actualitzacions. Si el nombre d'exemples no és divisible exactament per la mida del batch, l'últim batch és més petit i també compta com una actualització.

Un batch petit produeix més actualitzacions per època i calcula cada gradient amb menys exemples. Un batch gran produeix menys actualitzacions, però cada gradient utilitza més exemples. Cap mida és sempre millor.

## Learning rate i hiperparàmetres

El **learning rate** (`lr`) determina la mida del pas de cada actualització:

- Si és massa petit, l'aprenentatge pot ser molt lent i la loss encara pot baixar quan s'acaba l'entrenament.
- Si és massa gran, el model pot passar-se del mínim i fer que la loss oscil·li, pugi o es torni inestable.
- Un valor adequat permet que la loss baixi de manera útil i estable.

Els **hiperparàmetres** són decisions de configuració; el model no els aprèn com aprèn els pesos. A la pràctica es modifiquen:

- `lr`: mida del pas d'optimització.
- `batch`: nombre d'exemples processats en cada pas.
- `epoques`: nombre de passades per les dades d'entrenament.
- `n_train`: nombre d'exemples utilitzats per entrenar.
- L'arquitectura es manté fixa en una capa oculta de 64 neurones perquè els experiments se centrin en les dades i en els altres hiperparàmetres.

Per entendre l'efecte d'un hiperparàmetre, convé canviar-ne un cada vegada i mantenir els altres iguals.

## Mètriques, generalització i sobreajustament

Al final de cada època, la pràctica mesura les mètriques sota `torch.no_grad()`, que evita calcular gradients quan només s'està avaluant el model:

- **Loss:** mesura la qualitat de les prediccions tenint en compte la confiança del model. Més baixa acostuma a ser millor.
- **Accuracy** o **encert:** proporció d'imatges classificades correctament. Un encert del 90% vol dir que 9 de cada 10 imatges s'han classificat bé.

Les mètriques aporten informació diferent: l'accuracy només indica si la classe predita és correcta, mentre que la loss també reflecteix la confiança de la predicció.

**Generalitzar** vol dir obtenir bons resultats en exemples nous, no només en els utilitzats per entrenar. El **sobreajustament** (*overfitting*) apareix quan el model s'adapta massa a les dades d'entrenament. Un senyal habitual és que la loss d'entrenament continua baixant mentre la loss de validació deixa de baixar o comença a pujar. Es pot reduir, per exemple, utilitzant més dades, una xarxa menys gran o menys èpoques.
