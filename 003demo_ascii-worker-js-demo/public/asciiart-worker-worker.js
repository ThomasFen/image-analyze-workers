!function(e,t){if("object"==typeof exports&&"object"==typeof module)module.exports=t();else if("function"==typeof define&&define.amd)define([],t);else{var n=t();for(var r in n)("object"==typeof exports?exports:e)[r]=n[r]}}("undefined"!=typeof self?self:this,(function(){return(()=>{"use strict";var e={978:function(e,t,n){var r=this&&this.__awaiter||function(e,t,n,r){return new(n||(n=Promise))((function(o,a){function i(e){try{u(r.next(e))}catch(e){a(e)}}function s(e){try{u(r.throw(e))}catch(e){a(e)}}function u(e){var t;e.done?o(e.value):(t=e.value,t instanceof n?t:new n((function(e){e(t)}))).then(i,s)}u((r=r.apply(e,t||[])).next())}))},o=this&&this.__generator||function(e,t){var n,r,o,a,i={label:0,sent:function(){if(1&o[0])throw o[1];return o[1]},trys:[],ops:[]};return a={next:s(0),throw:s(1),return:s(2)},"function"==typeof Symbol&&(a[Symbol.iterator]=function(){return this}),a;function s(a){return function(s){return function(a){if(n)throw new TypeError("Generator is already executing.");for(;i;)try{if(n=1,r&&(o=2&a[0]?r.return:a[0]?r.throw||((o=r.return)&&o.call(r),0):r.next)&&!(o=o.call(r,a[1])).done)return o;switch(r=0,o&&(a=[2&a[0],o.value]),a[0]){case 0:case 1:o=a;break;case 4:return i.label++,{value:a[1],done:!1};case 5:i.label++,r=a[1],a=[0];continue;case 7:a=i.ops.pop(),i.trys.pop();continue;default:if(!((o=(o=i.trys).length>0&&o[o.length-1])||6!==a[0]&&2!==a[0])){i=0;continue}if(3===a[0]&&(!o||a[1]>o[0]&&a[1]<o[3])){i.label=a[1];break}if(6===a[0]&&i.label<o[1]){i.label=o[1],o=a;break}if(o&&i.label<o[2]){i.label=o[2],i.ops.push(a);break}o[2]&&i.ops.pop(),i.trys.pop();continue}a=t.call(e,i)}catch(e){a=[6,e],r=0}finally{n=o=0}if(5&a[0])throw a[1];return{value:a[0]?a[1]:void 0,done:!0}}([a,s])}}};Object.defineProperty(t,"__esModule",{value:!0});var a=n(970),i=self,s=99197/33405,u=function(e,t){return r(void 0,void 0,void 0,(function(){var n,r,a,i,u,c,l,f,d,p,h,m,v,y,g,x,I,b,w,M,C,T;return o(this,(function(o){for(n=t.asciiStr,r=t.fontSize,a=n.split(""),i=new OffscreenCanvas(e.width,e.height),(u=i.getContext("2d")).font=r+'px "Courier New", monospace',u.textBaseline="top",c=u.measureText(n),l=Math.floor(c.width/a.length),f=Math.ceil(e.width/l),d=Math.ceil(e.height/r),p=new OffscreenCanvas(f,d),(h=p.getContext("2d")).drawImage(e,0,0,f,d),m=h.getImageData(0,0,f,d),v=[],y=0;y<d;y++){for(g="",x=0;x<f;x++)I=4*(y*f+x),b=Math.max(0,Math.min(Math.floor((m.data[I+0]-128)*s)+128,255)),w=Math.max(0,Math.min(Math.floor((m.data[I+1]-128)*s)+128,255)),M=Math.max(0,Math.min(Math.floor((m.data[I+2]-128)*s)+128,255)),C=(.299*b+.587*w+.114*M)/255,T=a[a.length-1-Math.round(C*(a.length-1))],g+=T;v.push(g)}return[2,v]}))}))};onmessage=function(e){return r(void 0,void 0,void 0,(function(){var t,n,r,s;return o(this,(function(o){switch(o.label){case 0:return e.data.message!==a.WorkerCommand.INITIALIZE?[3,1]:(i.postMessage({message:a.WorkerResponse.INITIALIZED}),[3,4]);case 1:return e.data.message!==a.WorkerCommand.PREDICT?[3,3]:(t=e.data.image,n=e.data.uid,r=e.data.params,[4,u(t,r)]);case 2:return s=o.sent(),i.postMessage({message:a.WorkerResponse.PREDICTED,uid:n,lines:s}),t.close(),[3,4];case 3:console.log("not implemented"),o.label=4;case 4:return[2]}}))}))}},970:(e,t)=>{var n;Object.defineProperty(t,"__esModule",{value:!0}),t.AsciiFunctionType=t.WorkerResponse=t.WorkerCommand=void 0,t.WorkerCommand={INITIALIZE:"initialize",PREDICT:"predict"},t.WorkerResponse={INITIALIZED:"initialized",PREDICTED:"predicted"},(n=t.AsciiFunctionType||(t.AsciiFunctionType={}))[n.AsciiArt=0]="AsciiArt"}},t={};return function n(r){var o=t[r];if(void 0!==o)return o.exports;var a=t[r]={exports:{}};return e[r].call(a.exports,a,a.exports,n),a.exports}(978)})()}));