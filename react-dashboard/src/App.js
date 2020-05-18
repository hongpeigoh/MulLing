import React, { Component } from 'react';
import { Switch, Route, BrowserRouter } from 'react-router-dom';
import { Ripple } from '@progress/kendo-react-ripple';

import { About, AboutHead } from './components/about'
import { Home, HomeHead } from './components/home'
import { Contact, ContactHead } from'./components/contact'
import { Sandbox, SandboxHead, PopupWidget } from'./components/sandbox'
import Header from './components/navbar'

import '@progress/kendo-theme-material/dist/all.css';
import 'bootstrap-4-grid/css/grid.min.css';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isLoading: true,
      WidgetToggle: false
    };
    this.onKeyPressed = this.onKeyPressed.bind(this);
    this.handler = this.handler.bind(this);
  }
  componentDidMount() {
    document.getElementById("bootstrap-wrapper").focus();
  }
  onKeyPressed(e) {
    if (e.key === 'F7'){
      this.setState({
        WidgetToggle: !this.state.WidgetToggle
      });
    }
  }
  handler() {
    this.setState({
      WidgetToggle: !this.state.WidgetToggle
    });
  }

  render() {
    return (
      <Ripple>
        <BrowserRouter>
          
          <div 
            className="bootstrap-wrapper"
            id="bootstrap-wrapper"
            onKeyDown={this.onKeyPressed}
            tabIndex="0"
          >
            <Header />
            <HeaderMain />
            { this.state.WidgetToggle ? <PopupWidget handler={this.handler}/> : null }
            <Main />
          </div>
        </BrowserRouter>
      </Ripple>  
    )
  }
}

const HeaderMain = () => (
  <Switch>
    <Route exact path='/' component={HomeHead}></Route>
    <Route exact path='/about' component={AboutHead}></Route>
    <Route exact path='/contact' component={ContactHead}></Route>
    <Route exact path='/sandbox' component={SandboxHead}></Route>
  </Switch>
)

const Main = () => (
  <Switch>
    <Route exact path='/' component={Home}></Route>
    <Route exact path='/about' component={About}></Route>
    <Route exact path='/contact' component={Contact}></Route>
    <Route exact path='/sandbox' component={Sandbox}></Route>
  </Switch>
);

export default App;