import React, { Component, useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { Button } from '@progress/kendo-react-buttons';
import { Checkbox, NumericTextBox, } from '@progress/kendo-react-inputs';
import { DropDownList } from '@progress/kendo-react-dropdowns';
import withValueField from './withValueField.jsx';
import Loading from './widget.js'
import SearchProcess from '../images/searchprocess.png'

const DropDownListWithValueField = withValueField(DropDownList)

class HomeHead extends Component{
  render() {
    return (
      <div className="app-header" id="home-header">
        <div className="app-container container">
          <div className="row textcontent">
            <h1 className="center">MulLing</h1>
            <div className="col" id="searchbox">
              <Form/>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

class Home extends Component{
  render() {
    return (
      <div className='app-container container home width-70' id='app-container'>
        <div className="row" id="refresh-div" />
        <div className="row" id="input-to-results" />
        <div className="row" id="input-to-hidden-results" style={{display: "none"}}/>
        <div className="row textcontent">
          <h2>How does MulLing Search work?</h2>
          <img src={SearchProcess} alt="How does MulLing Search work?"/>
          <p><strong>Choose your filters: </strong>Vary the number of results to query, the input and output languages and the language model.</p>
          <p><strong>Query the database: </strong>Search from the corpora of articles from Singaporean news websites from the past few years.</p>
          <p><strong>Access the results: </strong>Expand to view the full articles and see related articles. Click the link to access an external source.</p>
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
    ReactDOM.unmountComponentAtNode(document.getElementById("input-to-results"));
    ReactDOM.unmountComponentAtNode(document.getElementById("input-to-hidden-results"));
    ReactDOM.render('', document.getElementById("refresh-div"));
    document.getElementById('searchbox').style.borderBottomWidth = "0px";
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
    this.handleValueChange = this.handleValueChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.toggleFilter = this.toggleFilter.bind(this);
    this.toggleResults = this.toggleResults.bind(this);
    this.toggleCollapse = this.toggleCollapse.bind(this);
    this.state = {
      lang: 'en',
      model: 'bai',
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
      isClust: true,
    }
    this.models = [
      {text: "Vector Addition Text", id: "baa"},
      {text: "Vector Addition Title", id: "meta"},
      {text: "TF-IDF Text", id: "bai"},
      {text: "TF-IDF Sentences", id: "senbai"},
      {text: "Bi-LSTM Text", id: "laser"},
      {text: "Bi-LSTM Title", id: "metalaser"},
      {text: "Bi-LSTM Sentences", id: "senlaser"}];
  }

  // componentDidUpdate(prevState) {
  //   for (const key in this.state) {
  //     if (this.state.key !== prevState.key) {
  //       this.setState({key: this.state.key});
  //     }
  //   }
  // }

  handleSubmit = e => {
    var self = this;
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
    const queryaddress = "http://localhost:5050/query_" + monolingual + "?q="+ encodedquery + "&model=" + this.state.model + "&lang=" + this.state.lang + "&k=" + this.state.k + suffix

    console.log(queryaddress);

    // Styling
    ReactDOM.render([
      <div className="col-6"><Button onClick={this.toggleResults} togglable={true} primary={true}>Cluster Results&nbsp;&nbsp;</Button></div>,
      <div className="col-6 right"><Refresh/></div> ], document.getElementById("refresh-div"));
    ReactDOM.render(<Loading />, document.getElementById("input-to-results"))

    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {      
      if (xhttp.readyState === 4 && xhttp.status === 200){
        // Add Results
        var PreChildren = [<PreSnippet key={0}/>];
        var MainChildren = [];
        var UnclusChildren = [];
        var PostChildren = [
          <span className="center" style={{width:"100%"}}>
            <Button icon="right" onClick={self.toggleCollapse}></Button>
          </span>];
        var myResults = JSON.parse(String(this.response)).allresults.map( x => x.split('\t'));

        // Add Clustered Results
        var ClusterLen = Math.max.apply(Math, myResults.map(e => e[2])) + 1; //Number of Clusters = Max of Clustering Labels
        for (let step = 0; step < ClusterLen; step++) {
          MainChildren.push([]);
        }

        for (const result of myResults){
          var fields = {
            rank: result[0],
            title: result[3],
            article: result[4],
            bestsentence: (result.length === 6 ? result[5] : null),
          };
          if ( MainChildren[result[2]].length === 0 ){
            MainChildren[result[2]].push(<span><div className="row"><h2>Cluster {parseInt(result[2],10)+1}:</h2></div></span>);
            MainChildren[result[2]].push(<Snippet key={String(fields.rank)} fields={fields} clust={true} isClust={false}/>);
          } else {
            if ( MainChildren[result[2]].length === 2 ){MainChildren[result[2]].push(<span><div className="row">Similar Results:</div></span>);}
            MainChildren[result[2]].push(<Snippet key={String(fields.rank)} fields={fields} clust={true} isClust={self.state.isClust}/>);
          }
          UnclusChildren.push(<Snippet key={String(fields.rank)} fields={fields} clust={false} isClust={self.state.isClust}/>);
        }

        ReactDOM.render(PreChildren.concat(UnclusChildren) , document.getElementById('input-to-results'));
        ReactDOM.render(PreChildren.concat(MainChildren.map((array, index)=>React.createElement( "span", { className:"cluster", id: "cluster" + String(index+1) }, array))).concat(PostChildren) , document.getElementById('input-to-hidden-results'));

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
  handleValueChange = (event) => {
    this.setState({
        model: event.target.value
    });
  }
  toggleFilter = e => {
    this.setState({ isCardView: !this.state.isCardView })
    if (document.getElementById("collapsible").style.maxHeight === "0px") {
      document.getElementById("collapsible").style.maxHeight = "400px";
    } else {
      document.getElementById("collapsible").style.maxHeight = "0px";
    }
  }
  toggleResults() {
    if (document.getElementById("input-to-results").style.display !== 'none') {
      document.getElementById("input-to-results").style.display = 'none';
      document.getElementById("input-to-hidden-results").style.display = 'flex';
    } else {
      document.getElementById("input-to-results").style.display = 'flex';
      document.getElementById("input-to-hidden-results").style.display = 'none';
    };
  }
  toggleCollapse() {
    this.setState({ isClust: !this.state.isClust })
  }

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
          <div className="col-12 col-md-4 col-xl-3">
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
          <div className="col-12 col-md-4 col-xl-3">
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
          <div className="col-12 col-md-4 col-xl-6">
            <div className="row">
              <div className="col-6 col-md-12 col-xl-6">
                <h4>Model</h4>
                <DropDownListWithValueField
                  data = {this.models}
                  textField = "text"
                  valueField = "id"
                  value = {this.state.model}
                  onChange = {this.handleValueChange}
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
  const [toggleMinimise, setToggleMinimise] = useState(Boolean(props.clust && props.isClust));
  const ddg_link = "https://duckduckgo.com/?q=!ducky+" + encodeURIComponent(props.fields.title)
  
  useEffect(() => {
    var changeResult = document.getElementById(`search-result-${props.fields.rank}${props.clust}`);
    changeResult.innerHTML = '<p>' + changeResult.innerText + '</p>';
  }, [display, props.fields.rank, props.clust]);
  
  return(
    ( props.clust && props.isClust && toggleMinimise ?
    <span className="result" onMouseEnter={() => setIsShown(true)} onMouseLeave={() => setIsShown(false)}>
      <div className="row">
        <small><button
          title={toggletext[0]}
          onClick={()=>setToggleMinimise(!toggleMinimise)}>
          {props.fields.title}</button></small>
      </div>
      <div className="row search-result" id={"search-result-" + props.fields.rank + props.clust} onClick={()=>setDisplay(1-display)} style={{display: "none"}}>
      </div>
      
    </span>
     :
    <span className="result" onMouseEnter={() => setIsShown(true)} onMouseLeave={() => setIsShown(false)}>
      <div className="row">
        <div className="col">
        <h2><a
          target="_blank"
          rel="noopener noreferrer"
          href={ddg_link}>
          {props.fields.title}</a></h2>
        </div>
        {( props.clust && props.isClust && !toggleMinimise ?
          <div className="col-2 right" style={{ paddingTop: "5px"}}>
            <span style={{ backgroundColor: "rgb(195,60,60"}}>
              <Button onClick={()=>setToggleMinimise(!toggleMinimise)}>Hide</Button>
            </span>
          </div>
          : null )}
        
        
      </div>
      <div className="row">
        <h5><a
          target="_blank"
          rel="noopener noreferrer"
          href={ddg_link}>
          {(ddg_link.length >= 90 ? ddg_link.substring(0,90) + '...' : ddg_link)}</a></h5>
      </div>
      <div className="row search-result" id={"search-result-" + props.fields.rank + props.clust} onClick={()=>setDisplay(1-display)}>
        {toggletext[display]}
      </div>
      <div className="row">
        <small>{props.fields.related}</small>
      </div>
      {isShown ? (
        <div className="row" style={{height: "36px"}}>
          {(toggletext[0]===toggletext[1] ? '' : <Button icon="hyperlink-open" onClick={()=>setDisplay(1-display)} look="bare" >{(display===0 ? 'Expand' : 'Collapse')} Article</Button>)}
        </div>
      ) : (
        <div className="row" style={{height: "36px"}}/>
      )}
      
    </span>
    )
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

export { Home, HomeHead };