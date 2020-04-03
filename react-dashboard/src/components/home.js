import React, { Component, useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { Button } from '@progress/kendo-react-buttons';
import { Checkbox, NumericTextBox } from '@progress/kendo-react-inputs';
import { DropDownList } from '@progress/kendo-react-dropdowns';
import Loading from './widget.js'

class Home extends Component{
  render() {
    return (
      <div className='home'>
        <div className="row" id="block-20"/>
        <div className="row" id="searchbox">
          <div className="col">
            <div className="row">
              <h1>Multilingual Information Retrieval (MulLing)</h1>
            </div>
            <Form/>
            <div className="row" id="refresh-div" />
            <div className="row" id="input-to-results" />
            </div>
        </div>
      </div>
    );
  }
}

class Refresh extends Component{
  constructor(props) {
    super(props);
    this.handleReload = this.handleReload.bind(this);
  }

  handleReload = e => {
    window.anim_searchbox.reverse();
    window.anim_blockcollapse.reverse();

    ReactDOM.unmountComponentAtNode(document.getElementById("input-to-results"));
    ReactDOM.render('', document.getElementById("refresh-div"));
  }

  render() {
    return(
      <span>
        <Button onClick={this.handleReload}>Clear Results</Button>
      </span>
    )
  }
}

class Form extends Component {
  constructor(props) {
    super(props);
    this.textInput = React.createRef();
    this.handleChange = this.handleChange.bind(this);
    this.handleCheckbox = this.handleCheckbox.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.toggleFilter = this.toggleFilter.bind(this);
    this.state = {
      lang: 'en',
      model: 'baa',
      k: 20,
      monolingual: true,
      disabled: true,
      lastselected: 'en',
      normalize: true,
      oen: true,
      ozh: true,
      oms: true,
      ota: true,
      isCardView: false,
    }
    this.models = ["baa", "meta", "bai", "senbai", "laser", "metalaser", "senlaser"]
    this.modelnames = ["Vector Addition Text", "Vector Addition Title", "TF-IDF Text", "TF-IDF Sentences", "Bi-LSTM Text", "Bi-LSTM Title", "Bi-LSTM Sentences"]
  }

  componentDidUpdate(prevState) {
    for (const key in this.state) {
      if (this.state.key !== prevState.key) {
        this.setState({key: this.state.key});
      }
    }
  }

  handleSubmit = e => {
    e.preventDefault();

    const monolingual = ( this.state.monolingual ? 'mono' : 'multi');
    const suffix = (
      this.state.monolingual ? '' :
      '&normalize=' + 
        String(this.state.normalize) + 
        '&oen=' + String(this.state.oen) + 
        '&ozh=' + String(this.state.ozh) + 
        '&oms=' + String(this.state.oms) + 
        '&ota=' + String(this.state.ota) );
    const encodedquery = encodeURIComponent(this.textInput.current.value)
    const queryaddress = "http://192.168.137.217:5050/query_" + monolingual + "?q="+ encodedquery + "&model=" + this.state.model + "&lang=" + this.state.lang + "&k=" + this.state.k + suffix

    console.log(queryaddress);

    // Styling
    ReactDOM.render(<Refresh/>, document.getElementById("refresh-div"));
    ReactDOM.render(<Loading />, document.getElementById("input-to-results"))

    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {      
      if (xhttp.readyState === 3 && xhttp.status === 200) {
        if (document.getElementById('searchbox').style.transform === "" | document.getElementById('searchbox').style.transform === "none") {
          window.anim = { fill: 'forwards', duration: 300 };
          window.anim_blockcollapse = document.getElementById('block-20').animate(
            [
              {height: '10vh'},
              {height: '0vh'}
            ], window.anim);
          window.anim_searchbox = document.getElementById('searchbox').animate(
            [
              {backgroundColor: 'rgba(255,255,255,0.6)', paddingTop: '5vw', borderRadius: '3vw', width: 'auto', transform: 'none', marginLeft: '0vw'},
              {backgroundColor: 'rgba(255,255,255,1)', paddingTop: '0', borderRadius: '0', width: '105vw', transform: 'translate3d(-20vw,0,0)', marginLeft: '5vw'}
            ], window.anim);
        }
      } else if(xhttp.readyState === 4 && xhttp.status === 200){
        // Add Results
        var Children = [<PreSnippet key={0}/>]
        var myResults = JSON.parse(String(this.response)).allresults.map( x => x.split('\t'));
        for (const result of myResults){
          let related = myResults.filter(function (e) {
            return e[2] === result[2] && e !== result;
          }).map(x=>x[0]);
          var relatedtext = (related.length > 3 ? 'Related to Articles: ' + related.slice(0,3).join(', ') : (related.length === 0 ? 'No Related Articles.' : 'Related to Articles: ' + related.join(', ')));

          var fields = {
            rank: result[0],
            title: result[3],
            article: result[4],
            bestsentence: (result.length === 6 ? result[5] : null),
            related: relatedtext
          };
          Children.push(<Snippet key={fields.rank} fields={fields} query={decodeURIComponent(encodedquery)}/>);
        }
        ReactDOM.render(Children, document.getElementById('input-to-results'));
      } else if(xhttp.readyState === 4 && xhttp.status === 500){
        ReactDOM.render(<ErrorSnippet status={xhttp.status}/>, document.getElementById('input-to-results'));
      }
    };
    xhttp.open("GET", queryaddress , true);
    xhttp.setRequestHeader('Access-Control-Allow-Origin','*');
    xhttp.send();
  };
  handleChange(e) {
    if (e.target.getAttribute('filter') === "lang") {
      this.setState({
        lang: e.target.value
      });
      if (e.target.value !== 'null'){
        this.setState({
          lastselected: e.target.value
        });
      }
    } else if (e.target.getAttribute('filter') === "k") {
      this.setState({
        k: e.target.value
      });
    }
  }
  handleCheckbox(e) {
    this.setState({
      monolingual: !this.state.monolingual,
      disabled: !this.state.disabled
    });
    if (this.state.lang === 'null') {
      this.setState({
        lang: this.state.lastselected
      });
    }
  }
  itemRender = (li, itemProps) => {
    const index = itemProps.index;
    const itemChildren = <span>{this.modelnames[index]} ({li.props.children})</span>;
    return React.cloneElement(li, li.props, itemChildren);
  }
  toggleFilter = e => {
    this.setState({ isCardView: !this.state.isCardView })
    if (document.getElementById("collapsible").style.maxHeight === "0px") {
      document.getElementById("collapsible").style.maxHeight = "400px";
    } else {
      document.getElementById("collapsible").style.maxHeight = "0px";
    }
  }
  valueRender = (element, value) => {
    const index = this.models.indexOf(value);
    const itemChildren = <span>{this.modelnames[index]} </span>;
    this.state.model = value;
    return React.cloneElement(element, element.props, itemChildren);
  }

  //{String(this.state.disabled).replace('false','(Unsupported)').replace('true','')}
  render() {
    return (
      <div>
      <div className="row" id="input-bar">
        
        <form className="input-form" onSubmit={this.handleSubmit}>
          <div className="row">
            <span id="input-text-area">
              <input id="input-text-field" className="input-bar" type="text" placeholder="Search..." ref={this.textInput}/>
            </span>
            <span className="right" style={{width:"36px"}}>
              <Button type="button" icon="close" look="bare" onClick={()=>{document.getElementById("input-text-field").value = ''}}/>
            </span>
            <span className="right" style={{width:"48px"}}>
              <Button type="button" style={{padding:"10px 0px 10px 16px"}} icon="filter" onClick={this.toggleFilter} look="bare">{ this.state.isCardView ? <span className="k-icon k-i-arrow-chevron-right" /> : <span className="k-icon k-i-arrow-chevron-down" />}</Button>
            </span>
            <span className="right" style={{width:"36px"}}>
              <Button icon="search" look="bare"/>
            </span>
          </div>
        </form>
      </div>
      <div className="row" id="collapsible">
          <div className="col-12 col-md-3">
            <h4>Input Language</h4>
            <div className="row">
              <div className="col-6 col-md-12 col-xl-6">
                <input type="radio" className="k-radio" filter="lang" value="en" checked={this.state.lang === "en"} onChange={this.handleChange}/>
                <label className="k-radio-label" htmlFor="r1">English</label><br/>
                <input type="radio" className="k-radio" filter="lang" value="zh" checked={this.state.lang === "zh"} onChange={this.handleChange}/>
                <label className="k-radio-label" htmlFor="r2">Chinese</label>
              </div>
              <div className="col-6 col-md-12 col-xl-6">
                <input type="radio" className="k-radio" filter="lang" value="ms" checked={this.state.lang === "ms"} onChange={this.handleChange}/>
                <label className="k-radio-label" htmlFor="r3">Malay</label><br/>
                <input type="radio" className="k-radio" filter="lang" value="ta" checked={this.state.lang === "ta"} onChange={this.handleChange} disabled={!this.state.disabled}/>
                <label className="k-radio-label" htmlFor="r4">Tamil</label>
                
              </div>
            </div>
            <div className="row" style={{marginTop: "3px"}}>
              <input type="radio" className="k-radio" filter="lang" value="null" checked={this.state.lang === "null"} onChange={this.handleChange} disabled={this.state.disabled}/>
              <label className="k-radio-label" htmlFor="r5">Unspecified</label>
            </div>
          </div>
          <div className="col-12 col-md-3">
            <h4>Output Language</h4>
            <div className="row">
              <div className="col-6 col-md-12 col-xl-6">
                <Checkbox disabled={this.state.disabled} checked={this.state.oen} onChange={(e) => this.setState({oen: !this.state.oen})} label={'English'}/><br/>
                <Checkbox disabled={this.state.disabled} checked={this.state.ozh} onChange={(e) => this.setState({ozh: !this.state.ozh})} label={'Chinese'}/><br/>
              </div>
              <div className="col-6 col-md-12 col-xl-6">
                <Checkbox disabled={this.state.disabled} checked={this.state.oms} onChange={(e) => this.setState({oms: !this.state.oms})} label={'Malay'}/><br/>
                <Checkbox disabled={this.state.disabled} checked={this.state.ota} onChange={(e) => this.setState({ota: !this.state.ota})} label={'Tamil'}/><br/>
              </div>
            </div>
          </div>
          <div className="col-12 col-md-3 col-xl-6">
            <div className="row">
              <div className="col-6 col-md-12 col-xl-6">
                <h4>Model</h4>
                <DropDownList
                  data = {this.models}
                  defaultValue = {this.models[2]}
                  itemRender = {this.itemRender}
                  valueRender = {this.valueRender}
                />
              </div>
              <div className="col-6 col-md-12 col-xl-6">
                <h4>Number of Results</h4>
                <NumericTextBox value={this.state.k} onChange={(e)=>this.setState({k: e.value})} max={1000} min ={1} width="140px"/><br/>
                <Checkbox onChange={this.handleCheckbox} label={'Multilingual Results'}/><br/>
                <Checkbox disabled={this.state.disabled} checked={this.state.normalize} onChange={(e) => this.setState({normalize: !this.state.normalize})} label={'Normalize Results'}/>
              </div>
            </div>
          </div>
      </div>
      </div>
    );
  }
}

function Snippet(props) {
  const toggletext =
  (props.fields.bestsentence == null
    ? [( props.fields.article.length > 275
        ? props.fields.article.substring(0,275) + ' ...'
        : props.fields.article ),
      props.fields.article.replace(/\n/g,"<br/><br/>")]
    : ['<b>Closest Sentence: </b>'+ props.fields.bestsentence,
    props.fields.article.replace(props.fields.bestsentence, '<b>'+props.fields.bestsentence+'</b>').replace(/\n/g,"<br/><br/>")] );
  const [display, setDisplay] = useState(0);
  const [isShown, setIsShown] = useState(false);
  const ddg_link = "https://duckduckgo.com/?q=!ducky+" + encodeURIComponent(props.fields.title)
  
  useEffect(() => {
    var changeResult = document.getElementById("search-result-" + props.fields.rank);
    changeResult.innerHTML = '<p>' + changeResult.innerText + '</p>';
  }, [display, props.fields.rank]);

  return(
    <span className="result" onMouseEnter={() => setIsShown(true)} onMouseLeave={() => setIsShown(false)}>
      <div className="row">
        <h2><a
          target="_blank"
          rel="noopener noreferrer"
          href={ddg_link}>
          {props.fields.title}</a></h2>
      </div>
      <div className="row">
        <h5><a
          target="_blank"
          rel="noopener noreferrer"
          href={ddg_link}>
          {(ddg_link.length >= 90 ? ddg_link.substring(0,90) + '...' : ddg_link)}</a></h5>
      </div>
      <div className="row search-result" id={"search-result-" + props.fields.rank} onClick={()=>setDisplay(1-display)}>
        {toggletext[display]}
      </div>
      <div className="row">
        <small>{props.fields.related}</small>
      </div>
      {isShown ? (
        <div className="row">
          {(toggletext[0]===toggletext[1] ? '' : <Button icon="hyperlink-open" onClick={()=>setDisplay(1-display)} look="bare" >{(display===0 ? 'Expand' : 'Collapse')} Article</Button>)}
        </div>
      ) : (
        <div className="row" style={{height: "36px"}}/>
      )}
      
    </span>
  )
}

const PreSnippet = () => (
  <span className="warning" style={{width:"100%"}}>
    <p style={{color:"red"}}>Warning: Links lead out using DuckDuckGo's proprietary I'm Feeling Lucky. Click at your own risk!</p>
  </span>
)

function ErrorSnippet(props) {
  return(
    <div className="row">
      <h2 style={{color:"red"}}>{(props.status === 500 ? 'Server Error' : 'Query Input Error'+ props.status)}</h2>
      <p style={{color:"red"}}>{(props.status === 500 ? 'Query could not be parsed. Please try again. Avoid using stopwords and out-of-dictionary words. Did you fill in an empty query?': 'Query could not be identified. Did you choose the correct input language?')}</p>
    </div>
  )
}

export default Home