import{r as d,j as e,d5 as P,v as s,F,R as v,w as E,aQ as S,d6 as L,d7 as R,d8 as a,d9 as w,da as z,b as A,db as j}from"./vendor-BC3OPQuM.js";import{S as C,j as k,Z as $,U as _,t as I,a4 as O}from"./vendor-arizeai-NjB3cZzD.js";import{E as T,L as D,a as N,P as G,h as M,M as U,b as m,D as B,d as q,c as J,e as K,f as W,g as H,T as Q,p as V,i as g,j as Y,k as Z,l as u,m as X,n as h,o as b,q as ee,r as re,s as ae,t as te,v as oe,A as ne,S as se,F as le}from"./pages-CnTvEGEN.js";import{b9 as ie,d as ce,R as de,ba as pe,bb as me}from"./components-Dte7_KRd.js";import"./vendor-three-DwGkEfCM.js";import"./vendor-recharts-BXLYwcXF.js";import"./vendor-codemirror-gE_JCOgX.js";(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))c(t);new MutationObserver(t=>{for(const o of t)if(o.type==="childList")for(const l of o.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&c(l)}).observe(document,{childList:!0,subtree:!0});function i(t){const o={};return t.integrity&&(o.integrity=t.integrity),t.referrerPolicy&&(o.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?o.credentials="include":t.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function c(t){if(t.ep)return;t.ep=!0;const o=i(t);fetch(t.href,o)}})();const f="arize-phoenix-feature-flags",p={__CLEAR__:!0};function ge(){const r=localStorage.getItem(f);if(!r)return p;try{const n=JSON.parse(r);return Object.assign({},p,n)}catch{return p}}const x=d.createContext(null);function ue(){const r=v.useContext(x);if(r===null)throw new Error("useFeatureFlags must be used within a FeatureFlagsProvider");return r}function he(r){const[n,i]=d.useState(ge()),c=t=>{localStorage.setItem(f,JSON.stringify(t)),i(t)};return e(x.Provider,{value:{featureFlags:n,setFeatureFlags:c},children:e(be,{children:r.children})})}function be(r){const{children:n}=r,{featureFlags:i,setFeatureFlags:c}=ue(),[t,o]=d.useState(!1);return P("ctrl+shift+f",()=>o(!0)),s(F,{children:[n,e(_,{type:"modal",isDismissable:!0,onDismiss:()=>o(!1),children:t&&e(C,{title:"Feature Flags",children:e(k,{height:"size-1000",padding:"size-100",children:Object.keys(i).map(l=>e($,{isSelected:i[l],onChange:y=>c({...i,[l]:y}),children:l},l))})})})]})}function fe(){return e(S,{styles:r=>E`
        body {
          background-color: var(--ac-global-color-grey-75);
          color: var(--ac-global-text-color-900);
          font-family: "Roboto";
          font-size: ${r.typography.sizes.medium.fontSize}px;
          margin: 0;
          #root,
          #root > div[data-overlay-container="true"],
          #root > div[data-overlay-container="true"] > .ac-theme {
            height: 100vh;
          }
        }

        /* Remove list styling */
        ul {
          display: block;
          list-style-type: none;
          margin-block-start: none;
          margin-block-end: 0;
          padding-inline-start: 0;
          margin-block-start: 0;
        }

        /* A reset style for buttons */
        .button--reset {
          background: none;
          border: none;
          padding: 0;
        }
        /* this css class is added to html via modernizr @see modernizr.js */
        .no-hiddenscroll {
          /* Works on Firefox */
          * {
            scrollbar-width: thin;
            scrollbar-color: var(--ac-global-color-grey-300)
              var(--ac-global-color-grey-400);
          }

          /* Works on Chrome, Edge, and Safari */
          *::-webkit-scrollbar {
            width: 14px;
          }

          *::-webkit-scrollbar-track {
            background: var(--ac-global-color-grey-100);
          }

          *::-webkit-scrollbar-thumb {
            background-color: var(--ac-global-color-grey-75);
            border-radius: 8px;
            border: 1px solid var(--ac-global-color-grey-300);
          }
        }

        :root {
          --px-blue-color: ${r.colors.arizeBlue};

          --px-flex-gap-sm: ${r.spacing.margin4}px;
          --px-flex-gap-sm: ${r.spacing.margin8}px;

          --px-section-background-color: ${r.colors.gray500};

          /* An item is a typically something in a list */
          --px-item-background-color: ${r.colors.gray800};
          --px-item-border-color: ${r.colors.gray600};

          --px-spacing-sm: ${r.spacing.padding4}px;
          --px-spacing-med: ${r.spacing.padding8}px;
          --px-spacing-lg: ${r.spacing.padding16}px;

          --px-border-radius-med: ${r.borderRadius.medium}px;

          --px-font-size-sm: ${r.typography.sizes.small.fontSize}px;
          --px-font-size-med: ${r.typography.sizes.medium.fontSize}px;
          --px-font-size-lg: ${r.typography.sizes.large.fontSize}px;

          --px-gradient-bar-height: 8px;

          --px-nav-collapsed-width: 45px;
          --px-nav-expanded-width: 200px;
        }

        .ac-theme--dark {
          --px-primary-color: #9efcfd;
          --px-primary-color--transparent: rgb(158, 252, 253, 0.2);
          --px-reference-color: #baa1f9;
          --px-reference-color--transparent: #baa1f982;
          --px-corpus-color: #92969c;
          --px-corpus-color--transparent: #92969c63;
        }
        .ac-theme--light {
          --px-primary-color: #00add0;
          --px-primary-color--transparent: rgba(0, 173, 208, 0.2);
          --px-reference-color: #4500d9;
          --px-reference-color--transparent: rgba(69, 0, 217, 0.2);
          --px-corpus-color: #92969c;
          --px-corpus-color--transparent: #92969c63;
        }
      `})}const xe=L(R(s(a,{path:"/",errorElement:e(T,{}),children:[e(a,{path:"/login",element:e(D,{})}),s(a,{element:e(N,{}),children:[e(a,{path:"/profile",handle:{crumb:()=>"profile"},element:e(G,{})}),e(a,{index:!0,loader:M}),s(a,{path:"/model",handle:{crumb:()=>"model"},element:e(U,{}),children:[e(a,{index:!0,element:e(m,{})}),e(a,{element:e(m,{}),children:e(a,{path:"dimensions",children:e(a,{path:":dimensionId",element:e(B,{}),loader:q})})}),e(a,{path:"embeddings",children:e(a,{path:":embeddingDimensionId",element:e(J,{}),loader:K,handle:{crumb:r=>r.embedding.name}})})]}),s(a,{path:"/projects",handle:{crumb:()=>"projects"},element:e(W,{}),children:[e(a,{index:!0,element:e(H,{})}),s(a,{path:":projectId",element:e(Q,{}),loader:V,handle:{crumb:r=>r.project.name},children:[e(a,{index:!0,element:e(g,{})}),e(a,{element:e(g,{}),children:e(a,{path:"traces/:traceId",element:e(Y,{})})})]})]}),s(a,{path:"/datasets",handle:{crumb:()=>"datasets"},children:[e(a,{index:!0,element:e(Z,{})}),s(a,{path:":datasetId",loader:u,handle:{crumb:r=>r.dataset.name},children:[s(a,{element:e(X,{}),loader:u,children:[e(a,{index:!0,element:e(h,{}),loader:b}),e(a,{path:"experiments",element:e(h,{}),loader:b}),e(a,{path:"examples",element:e(ee,{}),loader:re,children:e(a,{path:":exampleId",element:e(ae,{})})})]}),e(a,{path:"compare",handle:{crumb:()=>"compare"},loader:te,element:e(oe,{})})]})]}),e(a,{path:"/apis",element:e(ne,{}),handle:{crumb:()=>"APIs"}}),e(a,{path:"/settings",element:e(se,{}),handle:{crumb:()=>"Settings"}})]})]})),{basename:window.Config.basename});function ye(){return e(w,{router:xe})}function Pe(){return e(le,{children:e(ie,{children:e(Fe,{})})})}function Fe(){const{theme:r}=ce();return e(O,{theme:r,children:e(z,{theme:I,children:s(A.RelayEnvironmentProvider,{environment:de,children:[e(fe,{}),e(he,{children:e(pe,{children:e(d.Suspense,{children:e(me,{children:e(ye,{})})})})})]})})})}const ve=document.getElementById("root"),Ee=j.createRoot(ve);Ee.render(e(Pe,{}));
