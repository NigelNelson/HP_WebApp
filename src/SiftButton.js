class SiftButton extends React.Component {
    render() {
        return (
            <button type="button"
                    className="btn btn-primary w-25 border border-dark"
                    onClick={this.props.onClick}
                    style={{maxHWidth: '200px',
                        marginTop: '10px',
                        marginBottom: '15px '}}>
                Get Sift Points
            </button>
        )
    }
}