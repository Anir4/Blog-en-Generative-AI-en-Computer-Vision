# Generative AI en Computer Vision

> **Auteurs :**  
> Anir Idhmad  
> Mouad Lembarek  
> Houssam Hanzazi  
> Abdelali Hirri  
> **Date :** Avril 2026  
> **Tags :** `#GenerativeAI` `#ComputerVision` `#GAN` `#VAE` `#DiffusionModels` `#DeepLearning`

---

## Table des matières

1. [C'est quoi le Generative AI en Computer Vision ?](#1-cest-quoi-le-generative-ai-en-computer-vision-)
2. [Pourquoi utiliser le GenAI ?](#2-pourquoi-utiliser-le-genai-)
3. [Les architectures principales (ELI5)](#3-les-architectures-principales-eli5)
 - [GAN – Generative Adversarial Network](#31-gan--generative-adversarial-network)
 - [VAE – Variational Autoencoder](#32-vae--variational-autoencoder)
 - [Diffusion Models](#33-diffusion-models)
4. [Comparaison des trois architectures](#4-comparaison-des-trois-architectures)
5. [Références](#5-références)

---

## 1. C'est quoi le Generative AI en Computer Vision ?

### L'analogie du dessinateur magique

Imagine que tu as un ami qui a regardé **un million de photos de chats**. Maintenant, tu lui demandes de dessiner un chat qu'il n'a **jamais vu** — un chat violet avec des lunettes de soleil. Ton ami peut le faire parce qu'il a **appris les règles** de ce qu'est un chat.

Le **Generative AI en Computer Vision**, c'est exactement ça : des algorithmes capables de **créer des images, des vidéos ou des données visuelles** qui n'ont jamais existé, en ayant appris les patterns à partir de millions d'exemples réels.

---

### Définition formelle

> **Generative AI (GenAI)** désigne un ensemble de modèles d'apprentissage automatique capables de **générer de nouvelles données** (images, texte, son, vidéo) statistiquement similaires aux données d'entraînement.

En **Computer Vision (CV)**, cela se traduit par la capacité à :

- **Générer** des images réalistes depuis zéro (*text-to-image*)
- **Transformer** une image en une autre (*image-to-image*)
- **Compléter** ou **restaurer** des images dégradées (*inpainting*)
- **Synthétiser** des vidéos (*video generation*)

---

### Le cerveau du GenAI : apprendre la distribution des données

Mathématiquement, le problème revient à **apprendre la distribution probabiliste** des données réelles :

$$p_{data}(x)$$

où $x$ représente une image réelle. L'objectif du modèle génératif est d'apprendre un modèle $p_\theta(x)$ tel que :

$$p_\theta(x) \approx p_{data}(x)$$

Une fois cette distribution apprise, on peut **échantillonner** de nouvelles images :

$$x_{new} \sim p_\theta(x)$$

---

### Le paysage du Generative AI en CV

![GenAI Timeline](https://cdn.shopaccino.com/igmguru/images/generative-ai-evolution-68166659436944-1112253555507346.jpg)  
*Figure 1 — Vue d'ensemble du Generative AI (Source : researchgate)*

---

## 2. Pourquoi utiliser le GenAI ?

### 5 raisons concrètes

#### ① Résoudre le manque de données (Data Augmentation)

En Deep Learning, **plus on a de données, mieux c'est**. Mais collecter et annoter des données coûte cher et prend du temps.

**Exemple concret :** Un hôpital veut entraîner un modèle pour détecter des tumeurs rares. Il a seulement **50 images** de cas positifs. Avec un modèle génératif, il peut créer **50 000 images synthétiques** réalistes pour entraîner son détecteur.

```
Avant GenAI : 50 images → modèle peu performant (underfitting)
Après GenAI : 50 000 images → modèle robuste et généralisable
```

---

#### ② Créer du contenu visuel à grande échelle

Des entreprises comme Adobe, Canva ou Shutterstock utilisent le GenAI pour générer des visuels à la demande.

**Prompt exemple :** *"a futuristic city at sunset, photorealistic, 4K"* 
→ L'IA produit une image en quelques secondes.

---

#### ③ Restauration et amélioration d'images

- **Super-résolution** : transformer une image basse qualité en haute définition
- **Colorisation** : ajouter la couleur à des photos noir et blanc
- **Débruitage** : nettoyer des images médicales ou satellites

![Super Resolution Example](https://res.cloudinary.com/jerrick/image/upload/d_642250b563292b35f27461a7.png,f_jpg,fl_progressive,q_auto,w_1024/63315e3c2df1160020838173.jpg)  
*Figure 2 — Exemple de Super-Résolution par IA (Source : Wikimedia Commons)*

---

#### ④ Simulation et entraînement de robots / voitures autonomes

Tesla, Waymo, et d'autres utilisent des **environnements simulés** entièrement générés par IA pour entraîner leurs modèles de conduite autonome — millions de km virtuels avant un seul km réel.

---

#### ⑤ Art, Design et Créativité assistée

Des outils comme **Midjourney**, **DALL·E 3**, **Stable Diffusion** révolutionnent le design graphique, la mode, l'architecture et le cinéma.

---

### Tableau récapitulatif des cas d'usage

| Domaine | Cas d'usage | Modèle utilisé |
|---|---|---|
| Médical | Génération d'images IRM synthétiques | GAN, Diffusion |
| Automobile | Simulation de scènes de conduite | GAN |
| E-commerce | Essayage virtuel de vêtements | Diffusion |
| Cinéma | Deepfakes, effets spéciaux | GAN, VAE |
| Jeux vidéo | Génération de textures, niveaux | Diffusion |
| Satellite | Amélioration d'images géospatiales | VAE, GAN |

---

### 3.1 GAN — Generative Adversarial Network

#### L'analogie du faussaire et du détective

Imagine un **faussaire de billets de banque** (le Générateur) qui essaie de créer de faux billets aussi vrais que possible. En face, il y a un **détective** (le Discriminateur) dont le travail est de distinguer les vrais des faux billets.

- Le faussaire s'améliore pour tromper le détective.
- Le détective s'améliore pour détecter le faussaire.
- Résultat : après des millions de rounds, le faussaire crée des billets **parfaitement convaincants**.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/83/Generative_adversarial_network.svg" alt="GAN Architecture" width="500"/>
</p>  
*Figure 3 — Architecture GAN : Générateur vs Discriminateur (Source : Wikimedia Commons)*

---

#### Mathématiques du GAN

Le GAN résout un **jeu minimax à deux joueurs** :

$$\min_G \max_D \, V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

**Décomposons cette formule :**

| Terme | Signification |
|---|---|
| $\mathbb{E}_{x \sim p_{data}}[\log D(x)]$ | Le discriminateur doit donner une **haute probabilité** aux vraies images |
| $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$ | Le discriminateur doit donner une **basse probabilité** aux fausses images |
| $\min_G$ | Le générateur veut **minimiser** la capacité du discriminateur à détecter ses fakes |
| $\max_D$ | Le discriminateur veut **maximiser** sa capacité à distinguer vrai du faux |

**Point d'équilibre (Nash Equilibrium) :** L'entraînement converge (théoriquement) quand :

$$p_G(x) = p_{data}(x) \quad \Rightarrow \quad D(x) = \frac{1}{2} \text{ (ne peut plus distinguer)}$$

---

#### Architectures GAN populaires

| Architecture | Année | Innovation clé |
|---|---|---|
| **DCGAN** | 2015 | Convolutions dans G et D |
| **Pix2Pix** | 2017 | Image-to-image conditionnel |
| **CycleGAN** | 2017 | Traduction sans paires (cheval → zèbre) |
| **StyleGAN2** | 2020 | Contrôle du style à chaque résolution |
| **BigGAN** | 2018 | Génération haute résolution |

---

#### Avantages / Limites des GANs

| Avantages | Limites |
|---|---|
| Génération **ultra-rapide** (1 passe forward) | **Mode collapse** : G génère toujours les mêmes images |
| Résultats **très réalistes** (visages, scènes) | Entraînement **instable** et difficile |
| Flexible pour image-to-image | Peu de contrôle sur la **diversité** |
| Largement étudié (bibliothèques dispo) | Évaluation difficile (FID score subjectif) |

---

#### Exemple visuel : StyleGAN2

StyleGAN2 de NVIDIA génère des visages humains photoréalistes qui n'existent pas.

![StyleGAN Faces](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Woman_1.jpg/500px-Woman_1.jpg)  
*Figure 4 — Visages synthétiques générés par StyleGAN2 — thispersondoesnotexist.com*

### 3.2 VAE — Variational Autoencoder

#### L'analogie du résumé et de la reconstruction

Imagine que tu lis un roman de 1000 pages et tu dois le **résumer en 10 mots**. Ensuite, quelqu'un doit **réécrire le roman** à partir de tes 10 mots. C'est ce que fait le VAE !

- L'**Encodeur** compresse l'image en un petit "résumé" (espace latent).
- Le **Décodeur** reconstruit une image à partir de ce résumé.
- La magie : l'espace latent est **continu et ordonné**, donc on peut interpoler entre deux images.

![VAE Architecture](https://assets.ibm.com/is/image/ibm/variational-autoencoder-neural-network:16x9?fmt=png-alpha&dpr=on%2C1.25&wid=960&hei=540)  
*Figure 5 — Architecture VAE : Encodeur → Espace Latent → Décodeur (Source : Wikimedia Commons)*

---

#### Mathématiques du VAE

Le VAE optimise une **borne inférieure variationnelle (ELBO)** :

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction loss}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Régularisation KL}}$$

**Décomposons :**

**① Reconstruction Loss** : L'image reconstruite doit ressembler à l'originale

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] \approx -\| x - \hat{x} \|^2$$

**② Divergence KL** : Force l'espace latent à suivre une distribution normale $\mathcal{N}(0, I)$

$$D_{KL}(q_\phi(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**③ Reparametrization Trick** : Rend la backpropagation possible à travers l'échantillonnage

$$z = \mu + \sigma \odot \varepsilon \quad \text{où} \quad \varepsilon \sim \mathcal{N}(0, I)$$

---

#### La magie de l'espace latent

L'espace latent du VAE est **continu et interpolable** :

```
z_chat = [0.2, -0.5, 0.8, ...] ← Encoding d'un chat
z_chien = [0.9, 0.3, -0.1, ...] ← Encoding d'un chien

Interpolation :
z_milieu = 0.5 * z_chat + 0.5 * z_chien → Image hybride chat-chien 
```

Ce phénomène permet des applications comme :
- **Morphing** entre deux visages
- **Édition sémantique** (ajouter des lunettes, changer l'âge)
- **Génération contrôlée** par attributs

---

#### Avantages / Limites des VAEs

| Avantages | Limites |
|---|---|
| Espace latent **structuré et interprétable** | Images souvent **floues** (blur) |
| Entraînement **stable** | Qualité visuelle inférieure aux GANs |
| Idéal pour l'**interpolation** et l'édition | Compromis entre reconstruction et régularisation |
| Base pour beaucoup de modèles modernes (VQ-VAE, Stable Diffusion) | Plus complexe à implémenter que GAN |

---

### 3.3 Diffusion Models

#### L'analogie de la neige sur une photo

Imagine que tu prends une photo magnifique et que tu commences à ajouter de la **neige (bruit)** dessus, couche par couche, jusqu'à ce que la photo soit totalement blanche. 

Le modèle de diffusion apprend à faire **l'inverse** : partir de la neige complète et **retirer le bruit** étape par étape jusqu'à retrouver une belle image.

L'astuce géniale ? On peut partir de **neige aléatoire** et laisser le modèle créer une **toute nouvelle image** en retirant le bruit !

![Diffusion Process](https://towardsdatascience.com/wp-content/uploads/2022/05/1MLRitrnUdOy6rPtZfNwO4w.png)  
*Figure 6 — Processus Forward (ajout de bruit) et Reverse (débruitage) des Diffusion Models*

---

#### Mathématiques des Diffusion Models

##### Processus Forward

À chaque timestep $t$, on ajoute du bruit gaussien :

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \, \beta_t \mathbf{I})$$

**En une seule étape** (grâce à la propriété markovienne) :

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1-\bar{\alpha}_t) \mathbf{I})$$

où $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ est le **bruit cumulé** jusqu'à l'étape $t$.

Ce qui donne l'**échantillonnage direct** :

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1-\bar{\alpha}_t} \, \varepsilon \quad \text{où} \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})$$

---

##### Processus Reverse

Le modèle apprend $p_\theta(x_{t-1} | x_t)$ en prédisant le bruit :

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Fonction de perte simplifiée (DDPM)** :

$$\mathcal{L}_{simple} = \mathbb{E}_{x_0, \varepsilon, t} \left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]$$

En clair : le réseau U-Net $\varepsilon_\theta$ prédit le bruit $\varepsilon$ ajouté à l'étape $t$, et la loss mesure l'erreur entre le vrai bruit et le bruit prédit.

---

##### Conditional Generation (Text-to-Image)

Pour générer à partir d'un prompt texte, on utilise le **Classifier-Free Guidance (CFG)** :

$$\tilde{\varepsilon}_\theta(x_t, t, c) = \varepsilon_\theta(x_t, t, \emptyset) + w \cdot (\varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \emptyset))$$

- $c$ = embedding du texte (via CLIP)
- $w$ = guidance scale (plus $w$ est grand → plus fidèle au prompt)
- $\emptyset$ = pas de condition (génération non conditionnée)

---

## Architecture U-Net pour la prédiction du bruit

L’architecture U-Net constitue le cœur des modèles de diffusion modernes. Elle est conçue pour apprendre à prédire le bruit présent dans une image à un instant donné du processus de diffusion. Le modèle prend en entrée une image bruitée \(x_t\), un pas de temps \(t\) représentant le niveau de bruit, ainsi qu’une condition optionnelle \(c\), généralement issue d’un texte encodé dans les modèles conditionnels.

La structure de U-Net repose sur trois composantes principales. L’encodeur effectue une réduction progressive de la résolution spatiale tout en enrichissant les représentations internes. Cette phase permet d’extraire des caractéristiques pertinentes à différentes échelles grâce à l’utilisation de blocs résiduels et de mécanismes d’attention. Le bottleneck correspond à la représentation la plus compacte, où des mécanismes de self-attention capturent les dépendances globales, et où la cross-attention permet d’intégrer l’information textuelle.

Le décodeur reconstruit ensuite progressivement l’image en augmentant la résolution. Les skip connections jouent un rôle essentiel en transférant directement les informations de l’encodeur vers le décodeur, ce qui permet de préserver les détails fins et d’améliorer la qualité de la reconstruction. La sortie du modèle est une estimation du bruit \(\varepsilon_\theta\), ayant la même dimension que l’entrée. Le processus global repose sur l’apprentissage du débruitage itératif.

![Architecture U-Net](https://theaisummer.com/static/2c373d3667071700748bf451c4e62b78/7f018/long-skip-connection.jpg)  
*Figure 7 — Architecture U-Net avec connexions de saut (skip connections) entre l’encodeur et le décodeur*

---

## Stable Diffusion et la Latent Diffusion

Stable Diffusion introduit une amélioration majeure en réalisant le processus de diffusion dans un espace latent compressé plutôt que directement dans l’espace des pixels. Cette approche repose sur l’utilisation d’un autoencodeur variationnel (VAE) pour transformer une image de haute dimension (par exemple 512×512×3) en une représentation latente beaucoup plus compacte (64×64×4).

Dans ce cadre, l’image est d’abord encodée dans l’espace latent par le VAE. Le modèle de diffusion, basé sur U-Net, opère ensuite dans cet espace réduit afin de prédire et supprimer le bruit de manière itérative. Une fois le processus de débruitage terminé, le décodeur du VAE reconstruit l’image finale en haute résolution.

Cette stratégie permet de réduire considérablement le coût computationnel et la consommation mémoire, tout en maintenant une qualité visuelle élevée. En travaillant dans un espace compressé, le modèle devient plus efficace et accessible, ce qui a largement contribué à la démocratisation des modèles de diffusion dans des environnements disposant de ressources limitées.

![Latent Diffusion](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*vyzan-cdMVkpcRaVLkhQXA.png)  
*Figure 8 — Pipeline de la Latent Diffusion avec encodage VAE, diffusion dans l’espace latent et reconstruction*

---

#### Avantages / Limites des Diffusion Models

| Avantages | Limites |
|---|---|
| **Meilleure qualité** visuelle (SOTA en 2024-2026) | Génération **lente** (50-1000 étapes de débruitage) |
| **Haute diversité** des images générées | Coût computationnel **élevé** |
| Contrôle fin via **text-to-image** (CLIP) | Requiert beaucoup de **mémoire GPU** |
| Entraînement **stable** et théorique solide | Moins bon pour des domaines très spécifiques |
| Pas de mode collapse | Plus récent → moins mature |

---

## 4. Comparaison des trois architectures

### Comparaison côte à côte

| Critère | GAN | VAE | Diffusion Models |
|---|---|---|---|
| **Idée centrale** | Jeu adversarial faussaire/détective | Compression → reconstruction probabiliste | Débruitage progressif |
| **Date** | 2014 (Goodfellow) | 2013 (Kingma & Welling) | 2020 (Ho et al.) |
| **Vitesse de génération** | Très rapide (1 passe) | Rapide (1 passe) | Lent (N étapes) |
| **Qualité visuelle** | Très bonne | Floue parfois | SOTA |
| **Diversité** | Mode collapse possible | Bonne | Excellente |
| **Stabilité entraînement** | Instable | Stable | Stable |
| **Espace latent** | Non structuré | Structuré et interprétable | Partiel (Latent Diffusion) |
| **Contrôle** | Conditionnel (cGAN) | Par attributs latents | Text-to-image (CLIP) |
| **Mémoire GPU** | Faible | Faible | Élevée |
| **Applications phares** | Deepfakes, StyleGAN, CycleGAN | Interpolation, VAE-GAN, VQVAE | DALL·E, Midjourney, SD |

---

### Métriques d'évaluation communes

#### FID Score (Fréchet Inception Distance)

Mesure la **distance** entre la distribution des images réelles et générées :

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

- $\mu_r, \Sigma_r$ : moyenne et covariance des features des **vraies** images
- $\mu_g, \Sigma_g$ : moyenne et covariance des features des **images générées**
- **Plus le FID est bas → meilleure qualité** 

| Modèle | FID sur ImageNet 256×256 |
|---|---|
| GAN (BigGAN) | ~7.4 |
| VAE | ~30-50 |
| Diffusion (ADM) | ~3.9 |
| Diffusion + CFG | **~1.8** |

---

## Illustration visuelle du flux de données

![Comparaison GAN VAE Diffusion]([https://miro.medium.com/v2/resize:fit:1400/1*Jd6pGm7Fh9rKX7Q7lK8F2A.png](https://miro.medium.com/1*_5GpdejeOvt61ew4aPtT_g.png))  
*Figure 9 — Comparaison du flux de génération entre GAN, VAE et modèles de diffusion*

### GAN (Generative Adversarial Network)

Dans un GAN, le processus de génération est direct et s’effectue en une seule passe. Un vecteur latent \( z \), généralement échantillonné à partir d’une distribution normale \( \mathcal{N}(0, I) \), est fourni au générateur. Ce dernier produit immédiatement une image synthétique. Le modèle ne passe pas par une phase intermédiaire de reconstruction ou de raffinement progressif, ce qui rend la génération rapide mais parfois instable.

### VAE (Variational Autoencoder)

Le VAE repose sur un processus en deux étapes. Une image d’entrée \( x \) est d’abord encodée en deux vecteurs représentant la moyenne \( \mu \) et l’écart-type \( \sigma \). Un vecteur latent \( z \) est ensuite échantillonné à partir de cette distribution. Le décodeur reconstruit une approximation \( \hat{x} \) de l’image originale. Pour générer de nouvelles images, il suffit d’échantillonner un vecteur \( z \sim \mathcal{N}(0, I) \) et de le passer dans le décodeur.

### Modèles de diffusion

Les modèles de diffusion suivent une approche itérative. Le processus commence par un bruit pur \( x_T \sim \mathcal{N}(0, I) \). À travers une série d’étapes successives (souvent de l’ordre de 1000), un réseau de type U-Net prédit et retire progressivement le bruit pour obtenir des représentations intermédiaires \( x_{T-1}, \dots, x_1 \), jusqu’à produire une image finale \( x_0 \).

Contrairement aux GAN et aux VAE, la génération n’est pas instantanée mais progressive. Cette approche permet d’obtenir des résultats de haute qualité et plus stables, au prix d’un temps de calcul plus élevé.

---

### Cas d'usage par architecture

#### GAN — Meilleur pour :
- **Deepfakes et Face Swap** (manipulation de visages)
- **Super-résolution** (ESRGAN, Real-ESRGAN)
- **Transfert de style** (CycleGAN : photo → peinture)
- **Génération temps-réel** (jeux vidéo, streaming)

#### VAE — Meilleur pour :
- **Anomaly Detection** (l'image anormale se reconstruit mal)
- **Compression d'images** avec espace sémantique
- **VQ-VAE** → base de DALL·E original (2021)
- **Représentation apprise** pour downstream tasks

#### Diffusion — Meilleur pour :
- **Text-to-Image** (DALL·E 3, Midjourney v6, Stable Diffusion XL)
- **Inpainting** (remplir des zones manquantes)
- **Image editing** (Instruct Pix2Pix)
- **Medical imaging** (génération d'IRM, CT-scans)

---

## 5. Références

### Articles fondateurs (Papers)

1. **GAN original** — Goodfellow, I., et al. (2014). *Generative Adversarial Networks*. NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

2. **VAE original** — Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

3. **DDPM (Diffusion)** — Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

4. **DCGAN** — Radford, A., et al. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

5. **StyleGAN2** — Karras, T., et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN*. CVPR 2020. [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)

6. **CycleGAN** — Zhu, J.-Y., et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)

7. **Stable Diffusion (LDM)** — Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

8. **Classifier-Free Guidance** — Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

9. **DALL·E 2** — Ramesh, A., et al. (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents*. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)

10. **Score-Based Generative Models** — Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)

---

### Ressources pédagogiques

- **Andrej Karpathy** — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- **DeepMind x UCL Lectures** — [Deep Learning Course](https://www.deepmind.com/learning-resources)
- **Lilian Weng Blog** — [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- **Hugging Face** — [Diffusion Models Course](https://huggingface.co/learn/diffusion-courses)
- **Papers with Code** — [Image Generation Benchmarks](https://paperswithcode.com/task/image-generation)

---

### Outils & Bibliothèques

| Outil | Framework | Usage |
|---|---|---|
| [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | Python | Génération text-to-image locale |
| [Diffusers (HuggingFace)](https://github.com/huggingface/diffusers) | PyTorch | API pour diffusion models |
| [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) | PyTorch | Implémentations GAN classiques |
| [DALL·E API](https://platform.openai.com/docs/guides/images) | OpenAI | API text-to-image |
| [Midjourney](https://www.midjourney.com) | Discord Bot | Génération artistique |

---

> **Pour aller plus loin :** Le domaine évolue extrêmement vite. En 2025-2026, les **Video Diffusion Models** (Sora, Runway, Kling) et les **3D Generation Models** sont les nouvelles frontières du Generative AI en Computer Vision.

---
