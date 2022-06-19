class Switch extends React.Component {
    render() {
        return (
            <div className="form-check form-switch">
                <input className="form-check-input" type="checkbox" role="switch" id={this.props.id} onClick={this.props.handleClick}/>
                    <label className="form-check-label" htmlFor="flexSwitchCheckDefault">Default switch checkbox input</label>
            </div>
        )
    }
}